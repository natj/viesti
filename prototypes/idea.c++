#pragma once

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <iostream>


namespace GpuComm {

// ------------------------------------------------------------------
// Common Types
// ------------------------------------------------------------------

enum class BackendType {
    MPI,    // Uses CUDA-Aware MPI
    NCCL    // Uses NVIDIA NCCL
};

enum class DataType {
    FLOAT,
    INT,
    DOUBLE,
    // ... mappings to MPI_FLOAT / ncclFloat done internally
};


// ------------------------------------------------------------------
// The Abstract Request Object (Future)
// ------------------------------------------------------------------
// This unifies the "MPI Request" (which requires CPU waiting)
// and the "NCCL Stream" (which implies GPU waiting/event recording).
class Request {
public:
    virtual ~Request() = default;

    // Blocks the CPU until the operation is complete.
    // For MPI: Calls MPI_Wait.
    // For NCCL: Calls cudaStreamSynchronize (or cudaEventSynchronize).
    virtual void Wait() = 0;

    // Checks if complete without blocking.
    virtual bool Test() = 0;
};

// ------------------------------------------------------------------
// The Main Abstract Interface
// ------------------------------------------------------------------
class Communicator {

private:

    // process id (e.g., MPI rank)
    int rank_; 

    int size_;

public:
    virtual ~Communicator() = default;

    // --- Identification ---
    virtual int GetRank() const = 0;
    virtual int GetSize() const = 0;

    // --- Grouping Semantics (Crucial for NCCL P2P) ---
    // For MPI: no-op 
    // For NCCL: Maps to ncclGroupStart() / ncclGroupEnd().
    virtual void GroupStart() = 0;
    virtual void GroupEnd() = 0;

    // --- Point-to-Point Communication ---
    // Note: 'stream' is required for the interface, even if MPI uses it differently.
    // Returns a unique_ptr to a Request object for tracking.
    virtual std::unique_ptr<Request> send(
        const void* sendbuff, 
        size_t count, 
        DataType type, 
        int peer, 
        int tag, 
        cudaStream_t stream
    ) = 0;

    virtual std::unique_ptr<Request> recv(
        void* recvbuff, 
        size_t count, 
        DataType type, 
        int peer, 
        int tag, 
        cudaStream_t stream
    ) = 0;

    // --- Factory / Bootstrapping ---
    // The library must handle the MPI bootstrapping required for NCCL internally.
    static std::shared_ptr<Communicator> Create(BackendType backend, MPI_Comm bootstrapComm);
};

// ------------------------------------------------------------------
// Concrete Implementation: MPI
// ------------------------------------------------------------------
class MpiCommunicator : public Communicator {
private:
    MPI_Comm comm_;

    // Internal helper to map GpuComm::DataType to MPI_Datatype
    MPI_Datatype GetMpiType(DataType t);

public:
    // init mpi and ignore num_gpus_per_node 
    MpiCommunicator(MPI_Comm existingComm, int num_gpus_per_node);
    ~MpiCommunicator() override;

    int GetRank() const override; // returns rank_ because sub_rank_ is only needed by other backends
    int GetSize() const override;

    void GroupStart() override; // empty
    void GroupEnd() override;   // empty

    std::unique_ptr<Request> send(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override; // calls MPI_Isend
    std::unique_ptr<Request> recv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override; // calls MPI_Irecv
};

// Request Implementation for MPI
class MpiRequest : public Request {
private:
    MPI_Request mpi_req_;
public:
    MpiRequest(MPI_Request r) : mpi_req_(r) {}
    void Wait() override; // calls MPI_Wait(&mpi_req_, ...)
    bool Test() override; // calls MPI_Test(&mpi_req_, ...)
};


// ------------------------------------------------------------------
// Concrete Implementation: NCCL
// ------------------------------------------------------------------
class NcclCommunicator : public Communicator {
private:
    ncclComm_t nccl_comm_;
    MPI_Comm bootstrap_comm_; // Needed to exchange UniqueID during init

    // Additional rank assigned to this specific GPU context (NCCL logic)
    // Used for logic like "am I the root?" or calculating neighbors
    int sub_rank_; 

    // Internal helper to map GpuComm::DataType to ncclDataType_t
    ncclDataType_t GetNcclType(DataType t);

public:
    // Constructor handles the UniqueID exchange via MPI_Bcast
    NcclCommunicator(MPI_Comm bootstrapComm, int num_gpus_per_node) {
        MPI_Comm_rank(bootstrapComm, &rank_);

        // Calculate logical NCCL rank
        // Example logic for Multi-GPU setup:
        int local_gpu_id = GetLocalGpuId(); // e.g., 0, 1, 2, 3
        this->nccl_rank_ = (mpi_rank_ * num_gpus_per_node) + local_gpu_id;
        
        // Next, init NCCL using this.sub_rank_ ...

    }
    ~NcclCommunicator() override;

    // When the user asks "What is my rank?", return the NCCL rank because this is a GPU-communication library.
    // This is stored in sub_rank_
    int GetRank() const override 

    int GetSize() const override;

    void GroupStart() override; // calls ncclGroupStart
    void GroupEnd() override;   // calls ncclGroupEnd

    std::unique_ptr<Request> send(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override; //calls ncclSend
    std::unique_ptr<Request> recv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override; // calls ncclRecv
};




// Request Implementation for NCCL
// Since NCCL calls return immediately and queue on stream, 
// the "Request" effectively just marks a synchronization point.
class NcclRequest : public Request {
private:
    cudaEvent_t event_;
    cudaStream_t stream_;
public:
    NcclRequest(cudaStream_t s); // records event on creation
    ~NcclRequest();
    void Wait() override; // calls cudaEventSynchronize(event_)
    bool Test() override; // calls cudaEventQuery(event_)
};

} // namespace GpuComm
  


// Standard CUDA error check helper
#define CUDA_CHECK(call) { if(call != cudaSuccess) exit(1); }


// a simple test for both backends 
int main(int argc, char** argv) {

    // 1. Bootstrapping (Required for both)
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // 2. Set Device (Assume 1 GPU per rank)
    CUDA_CHECK(cudaSetDevice(rank));

    // 3. Create a CUDA Stream (REQUIRED for our Async API)
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 4. Allocate GPU Memory
    size_t count = 1024;
    float* d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, count * sizeof(float)));

    // Initialize data on GPU (Rank 0 = 1.0f, Rank 1 = 0.0f)
    if (rank == 0) {
        std::vector<float> h_data(count, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_buffer, h_data.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemset(d_buffer, 0, count * sizeof(float)));
    }
    

    // -------- TEST CASES START HERE ----------

    //A. Initialize Library with MPI Backend
    auto comm = GpuComm::Communicator::Create(
        GpuComm::BackendType::MPI, 
        MPI_COMM_WORLD
    );

    // B. Initialize Library with NCCL Backend
    // Internally: Rank 0 generates ID, Broadcasts it, and calls ncclCommInitRank
    //auto comm = GpuComm::Communicator::Create(
    //    GpuComm::BackendType::NCCL, 
    //    MPI_COMM_WORLD
    //);


    // The Portable Communication Logic
    // Even though MPI doesn't strictly need GroupStart, we use it 
    // because the code must be portable to NCCL.
    comm->GroupStart(); 

    std::unique_ptr<GpuComm::Request> req;

    if (rank == 0) {
        // Send from GPU memory
        req = comm->Isend(
            d_buffer, 
            count, 
            GpuComm::DataType::FLOAT, 
            1, // dest
            0, // tag
            stream
        );
    } else if (rank == 1) {
        // Receive into GPU memory
        req = comm->Irecv(
            d_buffer, 
            count, 
            GpuComm::DataType::FLOAT, 
            0, // source
            0, // tag
            stream
        );
    }

    comm->GroupEnd();

    // C. Wait for completion
    if (req) {
        req->Wait(); 
        std::cout << "Rank " << rank << " transfer complete." << std::endl;
    }


    // -------- TEST CASES END HERE ----------
      
    // Cleanup
    cudaFree(d_buffer);
    cudaStreamDestroy(stream);
    MPI_Finalize();
    return 0;
}












  
  
  
  
  
  
  
  
  
