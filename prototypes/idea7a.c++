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

// Abstract Communicator
class Communicator {
public:
    virtual ~Communicator() = default;

    // Factory method
    static std::unique_ptr<Communicator> create(Backend backend, MPI_Comm mpi_comm, cudaStream_t stream = nullptr);

    virtual int rank() = 0;
    virtual int size() = 0;

    // --- One-Sided (RMA) ---
    // Registers a local buffer to be accessible by other ranks via 'buffer_id'
    virtual void register_buffer(void* ptr, size_t size_bytes, int buffer_id) = 0;
    
    // Unregisters buffer
    virtual void deregister_buffer(int buffer_id) = 0;

    // Locks a specific remote buffer for access
    virtual void lock_buffer(int target_rank, int buffer_id) = 0;
    
    // Unlocks
    virtual void unlock_buffer(int target_rank, int buffer_id) = 0;

    // Put data from local 'src' to remote 'buffer_id' on 'target_rank'
    virtual std::unique_ptr<Request> put(const void* src, size_t count, DataType type, int target_rank, int buffer_id, int tag) = 0;

    // --- Two-Sided (Send/Recv) ---
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
    MPI_Comm comm_;
    int rank_;
    int size_;
    // Map buffer_id -> MPI_Win
    std::map<int, MPI_Win> windows_;

public:
    MpiCommunicator(MPI_Comm comm) : comm_(comm) {
        MPI_Comm_dup(comm, &comm_);
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }

    ~MpiCommunicator() {
        for (auto& pair : windows_) {
            MPI_Win_free(&pair.second);
        }
        MPI_Comm_free(&comm_);
    }

    int rank() override { return rank_; }
    int size() override { return size_; }

    void register_buffer(void* ptr, size_t size_bytes, int buffer_id) override {
        if (windows_.find(buffer_id) != windows_.end()) {
            throw std::runtime_error("Buffer ID already registered");
        }
        MPI_Win win;
        // Create window. Assuming CUDA-aware MPI, ptr is device pointer.
        MPI_Win_create(ptr, size_bytes, 1, MPI_INFO_NULL, comm_, &win);
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
        MPI_Isend(src, count, to_mpi_type(type), target_rank, tag, comm_, &req);
        return std::make_unique<MpiRequest>(req);
    }

    std::unique_ptr<Request> recv(void* dest, size_t count, DataType type, int source_rank, int tag) override {
        MPI_Request req;
        MPI_Irecv(dest, count, to_mpi_type(type), source_rank, tag, comm_, &req);
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
    int rank_;
    int size_;

public:
    NcclCommunicator(MPI_Comm mpi_comm, cudaStream_t stream) : stream_(stream) {
        MPI_Comm_rank(mpi_comm, &rank_);
        MPI_Comm_size(mpi_comm, &size_);

        ncclUniqueId id;
        if (rank_ == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);

        ncclCommInitRank(&nccl_comm_, size_, id, rank_);
    }

    ~NcclCommunicator() {
        ncclCommDestroy(nccl_comm_);
    }

    int rank() override { return rank_; }
    int size() override { return size_; }

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
std::unique_ptr<Communicator> Communicator::create(Backend backend, MPI_Comm mpi_comm, cudaStream_t stream) {
    if (backend == Backend::MPI) {
        return std::make_unique<MpiCommunicator>(mpi_comm);
    } else {
        return std::make_unique<NcclCommunicator>(mpi_comm, stream);
    }
}

} // namespace GpuComm


// ============================================================================
// MAIN APPLICATION
// ============================================================================

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Setup a stream for NCCL operations
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Select Backend: Switch to GpuComm::Backend::NV to test NCCL
  GpuComm::Backend backend = GpuComm::Backend::MPI; 
  // GpuComm::Backend backend = GpuComm::Backend::NV;

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

  // Create Communicator
  auto comm = GpuComm::Communicator::create(backend, MPI_COMM_WORLD, stream);

  // --------------------------------------------------
  // ONE SIDED TEST (Only works if Backend == MPI)
  // --------------------------------------------------
  if (backend == GpuComm::Backend::MPI) {
      int buf_id_A = 10;
      comm->register_buffer(buf_A, size_bytes, buf_id_A);
      
      MPI_Barrier(MPI_COMM_WORLD); // Ensure all registered before locking

      int target = 1; 
      
      // We lock the Specific Buffer on the Target Rank
      comm->lock_buffer(target, buf_id_A);

      if (rank == 0) {
        int tag = 0;
        // Put local buf_C into remote Rank 1's buf_A (ID 10)
        auto req = comm->put(buf_C, count, GpuComm::DataType::FLOAT, target, buf_id_A, tag);
        req->wait();
        std::cout << "Rank 0 Put data into Rank 1 buffer." << std::endl;
      }

      comm->unlock_buffer(target, buf_id_A);
      
      MPI_Barrier(MPI_COMM_WORLD);
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
  comm.reset(); // Destroy communicator before MPI_Finalize
  CUDA_CHECK(cudaFree(buf_A));
  CUDA_CHECK(cudaFree(buf_B));
  CUDA_CHECK(cudaFree(buf_C));
  CUDA_CHECK(cudaStreamDestroy(stream));

  MPI_Finalize();
  return 0;
}
