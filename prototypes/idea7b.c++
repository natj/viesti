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

enum class Backend {
    MPI,
    NV // NCCL
};

enum class DataType {
    FLOAT,
    DOUBLE,
    INT
};

// Abstract Request Object
class Request {
public:
    virtual ~Request() = default;
    virtual void wait() = 0;
};

// Abstract Communicator Parent Class
// Handles MPI init/finalize and common state (rank, size, comm)
class Communicator {
protected:
    MPI_Comm mpi_comm_;
    int rank_;
    int size_;
    bool owns_mpi_env_;

public:
    // Constructor initializes MPI if not already initialized
    Communicator(int* argc, char*** argv) {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            MPI_Init(argc, argv);
            owns_mpi_env_ = true;
        } else {
            owns_mpi_env_ = false;
        }

        // Duplicate world to ensure library isolation
        MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_);
        MPI_Comm_rank(mpi_comm_, &rank_);
        MPI_Comm_size(mpi_comm_, &size_);
    }

    virtual ~Communicator() {
        // Clean up our internal communicator
        MPI_Comm_free(&mpi_comm_);

        // Finalize MPI only if we were the ones who initialized it
        if (owns_mpi_env_) {
            int finalized;
            MPI_Finalized(&finalized);
            if (!finalized) {
                MPI_Finalize();
            }
        }
    }

    // Common Getters (Non-virtual)
    int rank() const { return rank_; }
    int size() const { return size_; }

    // Factory method
    static std::unique_ptr<Communicator> create(Backend backend, int* argc, char*** argv, cudaStream_t stream = nullptr);

    // --- Abstract API ---
    
    // One-Sided (RMA)
    virtual void register_buffer(void* ptr, size_t size_bytes, int buffer_id) = 0;
    virtual void deregister_buffer(int buffer_id) = 0;
    virtual void lock_buffer(int target_rank, int buffer_id) = 0;
    virtual void unlock_buffer(int target_rank, int buffer_id) = 0;
    virtual std::unique_ptr<Request> put(const void* src, size_t count, DataType type, int target_rank, int buffer_id, int tag) = 0;

    // Two-Sided (Send/Recv)
    virtual void start_sendrecv() = 0; 
    virtual void end_sendrecv() = 0;   
    virtual std::unique_ptr<Request> send(const void* src, size_t count, DataType type, int target_rank, int tag) = 0;
    virtual std::unique_ptr<Request> recv(void* dest, size_t count, DataType type, int source_rank, int tag) = 0;
};

// --------------------------------------------------------------------------
// Helper: Type Mapping
// --------------------------------------------------------------------------
MPI_Datatype to_mpi_type(DataType dt) {
    switch(dt) {
        case DataType::FLOAT: return MPI_FLOAT;
        case DataType::DOUBLE: return MPI_DOUBLE;
        case DataType::INT: return MPI_INT;
        default: return MPI_BYTE;
    }
}

size_t type_size(DataType dt) {
    switch(dt) {
        case DataType::FLOAT: return sizeof(float);
        case DataType::DOUBLE: return sizeof(double);
        case DataType::INT: return sizeof(int);
        default: return 1;
    }
}

// --------------------------------------------------------------------------
// MPI Backend Implementation
// --------------------------------------------------------------------------
class MpiRequest : public Request {
    MPI_Request req;
    bool active;
public:
    MpiRequest(MPI_Request r) : req(r), active(true) {}
    MpiRequest() : req(MPI_REQUEST_NULL), active(false) {} 
    
    void wait() override {
        if (active && req != MPI_REQUEST_NULL) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            active = false;
        }
    }
};

class MpiCommunicator : public Communicator {
    // Map buffer_id -> MPI_Win
    std::map<int, MPI_Win> windows_;

public:
    MpiCommunicator(int* argc, char*** argv) : Communicator(argc, argv) {
        // Parent constructor handles MPI Init and comm duplication
    }

    ~MpiCommunicator() override {
        for (auto& pair : windows_) {
            MPI_Win_free(&pair.second);
        }
    }

    void register_buffer(void* ptr, size_t size_bytes, int buffer_id) override {
        if (windows_.find(buffer_id) != windows_.end()) {
            throw std::runtime_error("Buffer ID already registered");
        }
        MPI_Win win;
        // Create window. Assuming CUDA-aware MPI, ptr is device pointer.
        // Use mpi_comm_ from parent class
        MPI_Win_create(ptr, size_bytes, 1, MPI_INFO_NULL, mpi_comm_, &win);
        windows_[buffer_id] = win;
    }

    void deregister_buffer(int buffer_id) override {
        auto it = windows_.find(buffer_id);
        if (it != windows_.end()) {
            MPI_Win_free(&it->second);
            windows_.erase(it);
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
        // MPI_Rput is Request-based put
        MPI_Rput(src, count, to_mpi_type(type), 
                 target_rank, 0, count, to_mpi_type(type), 
                 windows_.at(buffer_id), &req);
        return std::make_unique<MpiRequest>(req);
    }

    void start_sendrecv() override { /* No-op for MPI */ }
    void end_sendrecv() override { /* No-op for MPI */ }

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
// NCCL Backend Implementation
// --------------------------------------------------------------------------
class NcclRequest : public Request {
    cudaEvent_t event;
    cudaStream_t stream;
public:
    NcclRequest(cudaStream_t s) : stream(s) {
        cudaEventCreate(&event);
        cudaEventRecord(event, stream);
    }
    ~NcclRequest() {
        cudaEventDestroy(event);
    }
    void wait() override {
        cudaEventSynchronize(event);
    }
};

class NcclCommunicator : public Communicator {
    ncclComm_t nccl_comm_;
    cudaStream_t stream_;

public:
    NcclCommunicator(int* argc, char*** argv, cudaStream_t stream) : Communicator(argc, argv), stream_(stream) {
        // Parent constructor has already initialized MPI and duplicated mpi_comm_
        // Now use parent's mpi_comm_ to bootstrap NCCL

        ncclUniqueId id;
        if (rank_ == 0) ncclGetUniqueId(&id);
        
        // Use the parent's duplicated communicator
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm_);

        ncclCommInitRank(&nccl_comm_, size_, id, rank_);
    }

    ~NcclCommunicator() override {
        ncclCommDestroy(nccl_comm_);
    }

    void register_buffer(void* ptr, size_t size_bytes, int buffer_id) override {} 
    void deregister_buffer(int buffer_id) override {}
    void lock_buffer(int target_rank, int buffer_id) override {}
    void unlock_buffer(int target_rank, int buffer_id) override {}

    std::unique_ptr<Request> put(const void* src, size_t count, DataType type, int target_rank, int buffer_id, int tag) override {
        throw std::runtime_error("NCCL Backend does not support One-sided Put");
    }

    void start_sendrecv() override {
        ncclGroupStart();
    }

    void end_sendrecv() override {
        ncclGroupEnd();
    }

    std::unique_ptr<Request> send(const void* src, size_t count, DataType type, int target_rank, int tag) override {
        ncclSend(src, count * type_size(type), ncclChar, target_rank, nccl_comm_, stream_);
        return std::make_unique<NcclRequest>(stream_);
    }

    std::unique_ptr<Request> recv(void* dest, size_t count, DataType type, int source_rank, int tag) override {
        ncclRecv(dest, count * type_size(type), ncclChar, source_rank, nccl_comm_, stream_);
        return std::make_unique<NcclRequest>(stream_);
    }
};

// --------------------------------------------------------------------------
// Factory Implementation
// --------------------------------------------------------------------------
std::unique_ptr<Communicator> Communicator::create(Backend backend, int* argc, char*** argv, cudaStream_t stream) {
    if (backend == Backend::MPI) {
        return std::make_unique<MpiCommunicator>(argc, argv);
    } else {
        return std::make_unique<NcclCommunicator>(argc, argv, stream);
    }
}

} // namespace GpuComm


// ============================================================================
// MAIN APPLICATION
// ============================================================================

int main(int argc, char** argv) {
  
  // Note: MPI_Init is NO LONGER here. It is handled by the library.

  // Setup a stream for NCCL operations
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Select Backend
  GpuComm::Backend backend = GpuComm::Backend::MPI; 
  // GpuComm::Backend backend = GpuComm::Backend::NV;

  // Create Communicator FIRST (initializes MPI internally)
  // We pass pointers to argc/argv so MPI_Init can use them
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
      // Note: We cannot use MPI_Barrier(MPI_COMM_WORLD) easily here because 
      // the user doesn't own MPI_COMM_WORLD anymore technically, 
      // though for this single file test it would work.
      // Ideally, the library should expose a barrier, but we'll skip for now 
      // as the test logic is simple.
      
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
