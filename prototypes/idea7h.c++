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
#include <string>

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

struct Config {
    Backend rma_backend = Backend::MPI;
    Backend p2p_backend = Backend::MPI;
};

enum class DataType { FLOAT, DOUBLE, INT, UNKNOWN };

// Type Traits for Automatic Type Deduction
template<typename T> struct TypeTraits { static constexpr DataType value = DataType::UNKNOWN; };
template<> struct TypeTraits<float>    { static constexpr DataType value = DataType::FLOAT; };
template<> struct TypeTraits<double>   { static constexpr DataType value = DataType::DOUBLE; };
template<> struct TypeTraits<int>      { static constexpr DataType value = DataType::INT; };

namespace {
    size_t get_type_size(DataType dt) {
        switch(dt) {
            case DataType::FLOAT: return sizeof(float);
            case DataType::DOUBLE: return sizeof(double);
            case DataType::INT: return sizeof(int);
            default: return 1;
        }
    }

    void check_hip(hipError_t err, const char* file, int line) {
        if (err != hipSuccess) {
            throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err) + 
                                   " at " + file + ":" + std::to_string(line));
        }
    }
}
#define CHECK_HIP(cmd) check_hip(cmd, __FILE__, __LINE__)

// Abstract Request Object
class Request {
public:
    virtual ~Request() = default;
    virtual void wait() = 0;
};

// Abstract Communicator
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

    [[nodiscard]] static std::unique_ptr<Communicator> create(Config config, int* argc, char*** argv, hipStream_t stream = nullptr);

    // --- Core Virtual API (Type-Erased) ---
    virtual void register_buffer(void* ptr, size_t size_bytes, int buffer_id) = 0;
    virtual void deregister_buffer(int buffer_id) = 0;
    virtual void lock_buffer(int target_rank, int buffer_id) = 0;
    virtual void unlock_buffer(int target_rank, int buffer_id) = 0;
    
    [[nodiscard]] virtual std::unique_ptr<Request> put(const void* src, size_t count, DataType type, int target_rank, int buffer_id, int tag) = 0;
    
    virtual void start_sendrecv() = 0; 
    virtual void end_sendrecv() = 0;   
    
    [[nodiscard]] virtual std::unique_ptr<Request> send(const void* src, size_t count, DataType type, int target_rank, int tag) = 0;
    [[nodiscard]] virtual std::unique_ptr<Request> recv(void* dest, size_t count, DataType type, int source_rank, int tag) = 0;

    // --- Templated Convenience API (Modern C++ Wrapper) ---
    template <typename T>
    std::unique_ptr<Request> put(const T* src, size_t count, int target_rank, int buffer_id, int tag) {
        return put(src, count, TypeTraits<T>::value, target_rank, buffer_id, tag);
    }

    template <typename T>
    std::unique_ptr<Request> send(const T* src, size_t count, int target_rank, int tag) {
        return send(src, count, TypeTraits<T>::value, target_rank, tag);
    }

    template <typename T>
    std::unique_ptr<Request> recv(T* dest, size_t count, int source_rank, int tag) {
        return recv(dest, count, TypeTraits<T>::value, source_rank, tag);
    }
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
        // Note: MPI_Rput requires byte size if MPI_BYTE is used, but element count if a typed MPI_Datatype is used.
        // Our to_mpi_type returns typed MPI handles, so 'count' is correct.
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
// Composite Backend
// --------------------------------------------------------------------------
class CompositeCommunicator : public Communicator {
    std::unique_ptr<Communicator> rma_impl_;
    std::unique_ptr<Communicator> p2p_impl_;

public:
    CompositeCommunicator(std::unique_ptr<Communicator> rma, std::unique_ptr<Communicator> p2p, int* argc, char*** argv)
        : Communicator(argc, argv), rma_impl_(std::move(rma)), p2p_impl_(std::move(p2p)) {}

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

    void start_sendrecv() override { p2p_impl_->start_sendrecv(); }
    void end_sendrecv() override { p2p_impl_->end_sendrecv(); }
    
    std::unique_ptr<Request> send(const void* src, size_t count, DataType type, int target_rank, int tag) override {
        return p2p_impl_->send(src, count, type, target_rank, tag);
    }
    std::unique_ptr<Request> recv(void* dest, size_t count, DataType type, int source_rank, int tag) override {
        return p2p_impl_->recv(dest, count, type, source_rank, tag);
    }
};

// --------------------------------------------------------------------------
// Factory
// --------------------------------------------------------------------------
std::unique_ptr<Communicator> Communicator::create(Config config, int* argc, char*** argv, hipStream_t stream) {
    
    // Optimization: If identical, use direct implementation
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

    // Hybrid Construction
    std::unique_ptr<Communicator> rma;
    std::unique_ptr<Communicator> p2p;

    if (config.rma_backend == Backend::MPI) rma = std::make_unique<MpiCommunicator>(argc, argv);
#ifdef ENABLE_RCCL
    else rma = std::make_unique<RcclCommunicator>(argc, argv, stream);
#endif

    if (config.p2p_backend == Backend::MPI) p2p = std::make_unique<MpiCommunicator>(argc, argv);
#ifdef ENABLE_RCCL
    else p2p = std::make_unique<RcclCommunicator>(argc, argv, stream);
#endif

    return std::make_unique<CompositeCommunicator>(std::move(rma), std::move(p2p), argc, argv);
}

} // namespace GpuComm


// ============================================================================
// MAIN APPLICATION
// ============================================================================

int main(int argc, char** argv) {
  
  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  // Strategy Configuration
  GpuComm::Config config;
  config.rma_backend = GpuComm::Backend::MPI; 

#ifdef ENABLE_RCCL
  config.p2p_backend = GpuComm::Backend::RCCL;
  if(argc > 1) std::cout << "Mode: Hybrid MPI(RMA) + RCCL(P2P)" << std::endl;
#else
  config.p2p_backend = GpuComm::Backend::MPI;
  if(argc > 1) std::cout << "Mode: Pure MPI" << std::endl;
#endif

  auto comm = GpuComm::Communicator::create(config, &argc, &argv, stream);
  int rank = comm->rank();

  // Initialize data
  size_t count = 1024;
  size_t size_bytes = count * sizeof(float);
  float *buf_A, *buf_B, *buf_C;
  CHECK_HIP(hipMalloc(&buf_A, size_bytes)); 
  CHECK_HIP(hipMalloc(&buf_B, size_bytes)); 
  CHECK_HIP(hipMalloc(&buf_C, size_bytes)); 

  std::vector<float> h_A(count, 1.0f);
  std::vector<float> h_B(count, 2.0f);
  std::vector<float> h_C(count, rank == 0 ? 100.0f : 0.0f);
  
  CHECK_HIP(hipMemcpy(buf_A, h_A.data(), size_bytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(buf_B, h_B.data(), size_bytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(buf_C, h_C.data(), size_bytes, hipMemcpyHostToDevice));

  // --------------------------------------------------
  // ONE SIDED TEST (Via RMA Backend)
  // --------------------------------------------------
  int buf_id_A = 10;
  comm->register_buffer(buf_A, size_bytes, buf_id_A);
  comm->barrier();
  
  int target = 1; 
  comm->lock_buffer(target, buf_id_A);

  if (rank == 0) {
    // New: Templated API infers DataType::FLOAT automatically
    auto req = comm->put(buf_C, count, target, buf_id_A, 0);
    req->wait();
    std::cout << "Rank 0 Put data (RMA)." << std::endl;
  }
  comm->unlock_buffer(target, buf_id_A);
  comm->barrier();

  // --------------------------------------------------
  // SEND / RECV TEST (Via P2P Backend)
  // --------------------------------------------------
  comm->start_sendrecv();

  std::vector<std::unique_ptr<GpuComm::Request>> requests;
  if (rank == 0) {
    requests.push_back(comm->send(buf_C, count, 1, 666));
  } else if (rank == 1) {
    requests.push_back(comm->recv(buf_C, count, 0, 666));
  }

  comm->end_sendrecv();

  for(auto& req : requests) req->wait();
  
  if (rank == 1) std::cout << "Rank 1 finished recv (P2P)." << std::endl;

  // Cleanup
  comm.reset(); 
  CHECK_HIP(hipFree(buf_A));
  CHECK_HIP(hipFree(buf_B));
  CHECK_HIP(hipFree(buf_C));
  CHECK_HIP(hipStreamDestroy(stream));

  return 0;
}
