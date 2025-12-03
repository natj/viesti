#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <iostream>
#include <map>
#include <stdexcept>

#define CUDA_CHECK(cmd) { cudaError_t err = cmd; if (err != cudaSuccess) { printf("CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); } }

// ============================================================================
// LIBRARY DEFINITION
// ============================================================================

namespace GpuComm {

enum class Backend { MPI, NV };
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

    static std::unique_ptr<Communicator> create(Backend backend, int* argc, char*** argv, cudaStream_t stream = nullptr);

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
    ~MpiRequest() override { if(active) wait(); } // Safety: ensure wait is called
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
    using Communicator::Communicator; // Inherit constructor

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
// NCCL Backend
// --------------------------------------------------------------------------
class NcclRequest : public Request {
    cudaEvent_t event;
public:
    NcclRequest(cudaStream_t stream) {
        cudaEventCreate(&event);
        cudaEventRecord(event, stream);
    }
    ~NcclRequest() override { cudaEventDestroy(event); }
    void wait() override { cudaEventSynchronize(event); }
};

class NcclCommunicator : public Communicator {
    ncclComm_t nccl_comm_;
    cudaStream_t stream_;

public:
    NcclCommunicator(int* argc, char*** argv, cudaStream_t stream) : Communicator(argc, argv), stream_(stream) {
        ncclUniqueId id;
        if (rank_ == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm_);
        ncclCommInitRank(&nccl_comm_, size_, id, rank_);
    }

    ~NcclCommunicator() override { ncclCommDestroy(nccl_comm_); }

    void register_buffer(void*, size_t, int) override {} 
    void deregister_buffer(int) override {}
    void lock_buffer(int, int) override {}
    void unlock_buffer(int, int) override {}

    std::unique_ptr<Request> put(const void*, size_t, DataType, int, int, int) override {
        throw std::runtime_error("NCCL Backend does not support One-sided Put");
    }

    void start_sendrecv() override { ncclGroupStart(); }
    void end_sendrecv() override { ncclGroupEnd(); }

    std::unique_ptr<Request> send(const void* src, size_t count, DataType type, int target_rank, int) override {
        ncclSend(src, count * get_type_size(type), ncclChar, target_rank, nccl_comm_, stream_);
        return std::make_unique<NcclRequest>(stream_);
    }

    std::unique_ptr<Request> recv(void* dest, size_t count, DataType type, int source_rank, int) override {
        ncclRecv(dest, count * get_type_size(type), ncclChar, source_rank, nccl_comm_, stream_);
        return std::make_unique<NcclRequest>(stream_);
    }
};

std::unique_ptr<Communicator> Communicator::create(Backend backend, int* argc, char*** argv, cudaStream_t stream) {
    if (backend == Backend::MPI) return std::make_unique<MpiCommunicator>(argc, argv);
    else return std::make_unique<NcclCommunicator>(argc, argv, stream);
}

} // namespace GpuComm


// ============================================================================
// MAIN APPLICATION
// ============================================================================

int main(int argc, char** argv) {
  
  // Note: MPI_Init is handled by the library in create()

  // Setup a stream for NCCL operations
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Select Backend
  GpuComm::Backend backend = GpuComm::Backend::MPI; 
  // GpuComm::Backend backend = GpuComm::Backend::NV;

  // Create Communicator FIRST (initializes MPI internally)
  auto comm = GpuComm::Communicator::create(backend, &argc, &argv, stream);

  // Now we can use rank() to control logic
  int rank = comm->rank();
  int size = comm->size();

  // Initialize data on GPU
  size_t count = 1024;
  size_t size_bytes = count * sizeof(float);
  float *buf_A, *buf_B, *buf_C;
  CUDA_CHECK(cudaMalloc(&buf_A, size_bytes)); 
  CUDA_CHECK(cudaMalloc(&buf_B, size_bytes)); 
  CUDA_CHECK(cudaMalloc(&buf_C, size_bytes)); 

  // Init data
  std::vector<float> h_A(count, 1.0f);
  std::vector<float> h_B(count, 2.0f);
  std::vector<float> h_C(count, rank == 0 ? 100.0f : 0.0f);
  
  CUDA_CHECK(cudaMemcpy(buf_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf_B, h_B.data(), size_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf_C, h_C.data(), size_bytes, cudaMemcpyHostToDevice));

  // --------------------------------------------------
  // ONE SIDED TEST (Only works if Backend == MPI)
  // --------------------------------------------------
  if (backend == GpuComm::Backend::MPI) {
      int buf_id_A = 10;
      comm->register_buffer(buf_A, size_bytes, buf_id_A);
      
      // Barrier using internal communicator to ensure registration complete
      comm->barrier();
      
      int target = 1; 
      
      comm->lock_buffer(target, buf_id_A);

      if (rank == 0) {
        int tag = 0;
        // Put local buf_C into remote Rank 1's buf_A (ID 10)
        auto req = comm->put(buf_C, count, GpuComm::DataType::FLOAT, target, buf_id_A, tag);
        req->wait();
        std::cout << "Rank 0 Put data into Rank 1 buffer." << std::endl;
      }

      comm->unlock_buffer(target, buf_id_A);
  }

  // Ensure all ranks are ready before starting the next test
  comm->barrier();

  // --------------------------------------------------
  // SEND / RECV TEST
  // --------------------------------------------------
  
  comm->start_sendrecv();

  std::vector<std::unique_ptr<GpuComm::Request>> requests;
  int tag = 666;

  if (rank == 0) {
    requests.push_back(comm->send(buf_C, count, GpuComm::DataType::FLOAT, 1, tag));
  } else if (rank == 1) {
    requests.push_back(comm->recv(buf_C, count, GpuComm::DataType::FLOAT, 0, tag));
  }

  comm->end_sendrecv();

  for(auto& req : requests) {
      req->wait();
  }
  
  if (rank == 1) std::cout << "Rank 1 finished recv." << std::endl;

  // Cleanup
  comm.reset(); // Destroy communicator (Calls MPI_Finalize)
  CUDA_CHECK(cudaFree(buf_A));
  CUDA_CHECK(cudaFree(buf_B));
  CUDA_CHECK(cudaFree(buf_C));
  CUDA_CHECK(cudaStreamDestroy(stream));

  return 0;
}
