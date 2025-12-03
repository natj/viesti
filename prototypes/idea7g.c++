#include <mpi.h>
#ifdef ENABLE_RCCL
#include <rccl/rccl.h>
#endif
#include <hip/hip_runtime.h>
#include <vector>
#include <memory>
#include <iostream>
#include <map>
#include <stdexcept>

#define HIP_CHECK(cmd) { hipError_t err = cmd; if (err != hipSuccess) { printf("HIP Error: %s\n", hipGetErrorString(err)); exit(1); } }

// ============================================================================
// LIBRARY DEFINITION
// ============================================================================

namespace GpuComm {

enum class Backend { 
    MPI, 
#ifdef ENABLE_RCCL
    RCCL 
#endif
};

// Configuration Struct for Strategy Pattern
struct Config {
    Backend rma_backend = Backend::MPI;  // Backend for One-Sided (Put/Lock)
    Backend p2p_backend = Backend::MPI;  // Backend for Two-Sided (Send/Recv)
};

enum class DataType { FLOAT, DOUBLE, INT };

// Internal Helpers
namespace {
    size_t get_type_size(DataType dt) {
        switch(dt) {
            case DataType::FLOAT: return sizeof(float);
            case DataType::DOUBLE: return sizeof(double);
            case DataType::INT: return sizeof(int);
            default: return 1;
        }
    }
}

// Abstract Request Object
class Request {
public:
    virtual ~Request() = default;
    virtual void wait() = 0;
};

// Abstract Communicator Parent Class
class Communicator {
protected:
    MPI_Comm mpi_comm_;
    int rank_;
    int size_;
    bool owns_mpi_env_;

public:
    Communicator(int* argc, char*** argv) {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(argc, argv);
            owns_mpi_env_ = true;
        } else {
            owns_mpi_env_ = false;
        }
        MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_);
        MPI_Comm_rank(mpi_comm_, &rank_);
        MPI_Comm_size(mpi_comm_, &size_);
    }

    virtual ~Communicator() {
        MPI_Comm_free(&mpi_comm_);
        if (owns_mpi_env_) {
            int finalized;
            MPI_Finalized(&finalized);
            if (!finalized) MPI_Finalize();
        }
    }

    int rank() const { return rank_; }
    int size() const { return size_; }
    void barrier() { MPI_Barrier(mpi_comm_); }

    static std::unique_ptr<Communicator> create(Config config, int* argc, char*** argv, hipStream_t stream = nullptr);

    // API
    virtual void register_buffer(void* ptr, size_t size_bytes, int buffer_id) = 0;
    virtual void deregister_buffer(int buffer_id) = 0;
    virtual void lock_buffer(int target_rank, int buffer_id) = 0;
    virtual void unlock_buffer(int target_rank, int buffer_id) = 0;
    virtual std::unique_ptr<Request> put(const void* src, size_t count, DataType type, int target_rank, int buffer_id, int tag) = 0;

    virtual void start_sendrecv() = 0; 
    virtual void end_sendrecv() = 0;   
    virtual std::unique_ptr<Request> send(const void* src, size_t count, DataType type, int target_rank, int tag) = 0;
    virtual std::unique_ptr<Request> recv(void* dest, size_t count, DataType type, int source_rank, int tag) = 0;
};

// --------------------------------------------------------------------------
// MPI Backend
// --------------------------------------------------------------------------
class MpiRequest : public Request {
    MPI_Request req;
    bool active;
public:
    MpiRequest(MPI_Request r) : req(r), active(true) {}
    ~MpiRequest() override { if(active) wait(); } 
    void wait() override {
        if (active && req != MPI_REQUEST_NULL) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            active = false;
        }
    }
};

class MpiCommunicator : public Communicator {
    std::map<int, MPI_Win> windows_;

    static MPI_Datatype to_mpi_type(DataType dt) {
        switch(dt) {
            case DataType::FLOAT: return MPI_FLOAT;
            case DataType::DOUBLE: return MPI_DOUBLE;
            case DataType::INT: return MPI_INT;
            default: return MPI_BYTE;
        }
    }

public:
    using Communicator::Communicator; 

    ~MpiCommunicator() override {
        for (auto& pair : windows_) MPI_Win_free(&pair.second);
    }

    void register_buffer(void* ptr, size_t size_bytes, int buffer_id) override {
        if (windows_.count(buffer_id)) throw std::runtime_error("Buffer ID exists");
        MPI_Win win;
        MPI_Win_create(ptr, size_bytes, 1, MPI_INFO_NULL, mpi_comm_, &win);
        windows_[buffer_id] = win;
    }

    void deregister_buffer(int buffer_id) override {
        if (windows_.count(buffer_id)) {
            MPI_Win_free(&windows_[buffer_id]);
            windows_.erase(buffer_id);
        }
    }

    void lock_buffer(int target_rank, int buffer_id) override {
        MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, windows_.at(buffer_id));
    }

    void unlock_buffer(int target_rank, int buffer_id) override {
        MPI_Win_unlock(target_rank, windows_.at(buffer_id));
    }

    std::unique_ptr<Request> put(const void* src, size_t count, DataType type, int target_rank, int buffer_id, int tag) override {
        MPI_Request req;
        MPI_Rput(src, count, to_mpi_type(type), target_rank, 0, count, to_mpi_type(type), windows_.at(buffer_id), &req);
        return std::make_unique<MpiRequest>(req);
    }

    void start_sendrecv() override {}
    void end_sendrecv() override {}

    std::unique_ptr<Request> send(const void* src, size_t count, DataType type, int target_rank, int tag) override {
        MPI_Request req;
        MPI_Isend(src, count, to_mpi_type(type), target_rank, tag, mpi_comm_, &req);
        return std::make_unique<MpiRequest>(req);
    }

    std::unique_ptr<Request> recv(void* dest, size_t count, DataType type, int source_rank, int tag) override {
        MPI_Request req;
        MPI_Irecv(dest, count, to_mpi_type(type), source_rank, tag, mpi_comm_, &req);
        return std::make_unique<MpiRequest>(req);
    }
};

// --------------------------------------------------------------------------
// RCCL Backend
// --------------------------------------------------------------------------
#ifdef ENABLE_RCCL
class RcclRequest : public Request {
    hipEvent_t event;
public:
    RcclRequest(hipStream_t stream) {
        hipEventCreate(&event);
        hipEventRecord(event, stream);
    }
    ~RcclRequest() override { hipEventDestroy(event); }
    void wait() override { hipEventSynchronize(event); }
};

class RcclCommunicator : public Communicator {
    ncclComm_t nccl_comm_;
    hipStream_t stream_;

public:
    RcclCommunicator(int* argc, char*** argv, hipStream_t stream) : Communicator(argc, argv), stream_(stream) {
        ncclUniqueId id;
        if (rank_ == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm_);
        ncclCommInitRank(&nccl_comm_, size_, id, rank_);
    }

    ~RcclCommunicator() override { ncclCommDestroy(nccl_comm_); }

    void register_buffer(void*, size_t, int) override {} 
    void deregister_buffer(int) override {}
    void lock_buffer(int, int) override {}
    void unlock_buffer(int, int) override {}

    std::unique_ptr<Request> put(const void*, size_t, DataType, int, int, int) override {
        throw std::runtime_error("RCCL Backend does not support One-sided Put");
    }

    void start_sendrecv() override { ncclGroupStart(); }
    void end_sendrecv() override { ncclGroupEnd(); }

    std::unique_ptr<Request> send(const void* src, size_t count, DataType type, int target_rank, int) override {
        ncclSend(src, count * get_type_size(type), ncclChar, target_rank, nccl_comm_, stream_);
        return std::make_unique<RcclRequest>(stream_);
    }

    std::unique_ptr<Request> recv(void* dest, size_t count, DataType type, int source_rank, int) override {
        ncclRecv(dest, count * get_type_size(type), ncclChar, source_rank, nccl_comm_, stream_);
        return std::make_unique<RcclRequest>(stream_);
    }
};
#endif

// --------------------------------------------------------------------------
// Composite Backend (For Hybrid Configurations)
// --------------------------------------------------------------------------
class CompositeCommunicator : public Communicator {
    std::unique_ptr<Communicator> rma_impl_;
    std::unique_ptr<Communicator> p2p_impl_;

public:
    CompositeCommunicator(std::unique_ptr<Communicator> rma, std::unique_ptr<Communicator> p2p, int* argc, char*** argv)
        : Communicator(argc, argv), rma_impl_(std::move(rma)), p2p_impl_(std::move(p2p)) {}

    // Delegate RMA operations to rma_impl_
    void register_buffer(void* ptr, size_t size_bytes, int buffer_id) override {
        rma_impl_->register_buffer(ptr, size_bytes, buffer_id);
    }
    void deregister_buffer(int buffer_id) override {
        rma_impl_->deregister_buffer(buffer_id);
    }
    void lock_buffer(int target_rank, int buffer_id) override {
        rma_impl_->lock_buffer(target_rank, buffer_id);
    }
    void unlock_buffer(int target_rank, int buffer_id) override {
        rma_impl_->unlock_buffer(target_rank, buffer_id);
    }
    std::unique_ptr<Request> put(const void* src, size_t count, DataType type, int target_rank, int buffer_id, int tag) override {
        return rma_impl_->put(src, count, type, target_rank, buffer_id, tag);
    }

    // Delegate P2P operations to p2p_impl_
    void start_sendrecv() override {
        p2p_impl_->start_sendrecv();
    }
    void end_sendrecv() override {
        p2p_impl_->end_sendrecv();
    }
    std::unique_ptr<Request> send(const void* src, size_t count, DataType type, int target_rank, int tag) override {
        return p2p_impl_->send(src, count, type, target_rank, tag);
    }
    std::unique_ptr<Request> recv(void* dest, size_t count, DataType type, int source_rank, int tag) override {
        return p2p_impl_->recv(dest, count, type, source_rank, tag);
    }
};

// --------------------------------------------------------------------------
// Factory Logic
// --------------------------------------------------------------------------
std::unique_ptr<Communicator> Communicator::create(Config config, int* argc, char*** argv, hipStream_t stream) {
    
    // Optimization: If backends are identical, don't wrap them in a composite
    if (config.rma_backend == config.p2p_backend) {
        if (config.rma_backend == Backend::MPI) {
            return std::make_unique<MpiCommunicator>(argc, argv);
        }
#ifdef ENABLE_RCCL
        else if (config.rma_backend == Backend::RCCL) {
            return std::make_unique<RcclCommunicator>(argc, argv, stream);
        }
#endif
    }

    // If backends differ, create components and wrap in Composite
    std::unique_ptr<Communicator> rma;
    std::unique_ptr<Communicator> p2p;

    // 1. Create RMA Backend
    if (config.rma_backend == Backend::MPI) {
        rma = std::make_unique<MpiCommunicator>(argc, argv);
    } 
#ifdef ENABLE_RCCL
    else {
        rma = std::make_unique<RcclCommunicator>(argc, argv, stream);
    }
#endif

    // 2. Create P2P Backend
    if (config.p2p_backend == Backend::MPI) {
        p2p = std::make_unique<MpiCommunicator>(argc, argv);
    }
#ifdef ENABLE_RCCL
    else {
        p2p = std::make_unique<RcclCommunicator>(argc, argv, stream);
    }
#endif

    return std::make_unique<CompositeCommunicator>(std::move(rma), std::move(p2p), argc, argv);
}

} // namespace GpuComm


// ============================================================================
// MAIN APPLICATION
// ============================================================================

int main(int argc, char** argv) {
  
  // Setup a stream for RCCL operations
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // --------------------------------------------------------------------------
  // Configuration: Strategy Pattern
  // --------------------------------------------------------------------------
  GpuComm::Config config;
  
  // Use MPI for One-Sided (RMA) operations
  config.rma_backend = GpuComm::Backend::MPI; 

  // Use RCCL for Send/Recv operations (if enabled), otherwise fallback to MPI
#ifdef ENABLE_RCCL
  config.p2p_backend = GpuComm::Backend::RCCL;
  if(argc > 1) std::cout << "Using Hybrid MPI (RMA) + RCCL (P2P) mode" << std::endl;
#else
  config.p2p_backend = GpuComm::Backend::MPI;
  if(argc > 1) std::cout << "Using MPI for both RMA and P2P" << std::endl;
#endif

  // Create Communicator with Configuration
  auto comm = GpuComm::Communicator::create(config, &argc, &argv, stream);

  // Now we can use rank() to control logic
  int rank = comm->rank();
  int size = comm->size();

  // Initialize data on GPU
  size_t count = 1024;
  size_t size_bytes = count * sizeof(float);
  float *buf_A, *buf_B, *buf_C;
  HIP_CHECK(hipMalloc(&buf_A, size_bytes)); 
  HIP_CHECK(hipMalloc(&buf_B, size_bytes)); 
  HIP_CHECK(hipMalloc(&buf_C, size_bytes)); 

  // Init data
  std::vector<float> h_A(count, 1.0f);
  std::vector<float> h_B(count, 2.0f);
  std::vector<float> h_C(count, rank == 0 ? 100.0f : 0.0f);
  
  HIP_CHECK(hipMemcpy(buf_A, h_A.data(), size_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(buf_B, h_B.data(), size_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(buf_C, h_C.data(), size_bytes, hipMemcpyHostToDevice));

  // --------------------------------------------------
  // ONE SIDED TEST (Uses rma_backend = MPI)
  // --------------------------------------------------
  
  int buf_id_A = 10;
  // This registers buffer in the MPI Window (delegated by Composite)
  comm->register_buffer(buf_A, size_bytes, buf_id_A);
  
  comm->barrier();
  
  int target = 1; 
  
  comm->lock_buffer(target, buf_id_A);

  if (rank == 0) {
    int tag = 0;
    // Uses MPI_Rput internally
    auto req = comm->put(buf_C, count, GpuComm::DataType::FLOAT, target, buf_id_A, tag);
    req->wait();
    std::cout << "Rank 0 Put data into Rank 1 buffer (Via MPI)." << std::endl;
  }

  comm->unlock_buffer(target, buf_id_A);

  // Ensure all ranks are ready before starting the next test
  comm->barrier();

  // --------------------------------------------------
  // SEND / RECV TEST (Uses p2p_backend = RCCL)
  // --------------------------------------------------
  
  comm->start_sendrecv(); // Calls ncclGroupStart

  std::vector<std::unique_ptr<GpuComm::Request>> requests;
  int tag = 666;

  if (rank == 0) {
    // Uses ncclSend internally
    requests.push_back(comm->send(buf_C, count, GpuComm::DataType::FLOAT, 1, tag));
  } else if (rank == 1) {
    // Uses ncclRecv internally
    requests.push_back(comm->recv(buf_C, count, GpuComm::DataType::FLOAT, 0, tag));
  }

  comm->end_sendrecv(); // Calls ncclGroupEnd

  for(auto& req : requests) {
      req->wait(); // Waits on hipEvent
  }
  
  if (rank == 1) std::cout << "Rank 1 finished recv (Via RCCL)." << std::endl;

  // Cleanup
  comm.reset(); 
  HIP_CHECK(hipFree(buf_A));
  HIP_CHECK(hipFree(buf_B));
  HIP_CHECK(hipFree(buf_C));
  HIP_CHECK(hipStreamDestroy(stream));

  return 0;
}
