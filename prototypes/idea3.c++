#pragma once
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <memory>
#include <iostream>
#include <vector>

namespace GpuComm {

// Enums
enum class Backend { MPI, NCCL };
enum class DataType { FLOAT, INT, DOUBLE };

// ------------------------------------------------------------------
// Abstract Interfaces
// ------------------------------------------------------------------

struct Request {
    virtual ~Request() = default;
    virtual void Wait() = 0;
};

class Communicator {
protected:
    int rank_ = 0;
    int size_ = 1;

public:
    virtual ~Communicator() = default;
    
    // Simplification: Non-virtual getters
    int rank() const { return rank_; }
    int size() const { return size_; }

    virtual void groupStart() = 0;
    virtual void groupEnd() = 0;

    virtual std::unique_ptr<Request> send(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) = 0;
    virtual std::unique_ptr<Request> recv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) = 0;

    static std::shared_ptr<Communicator> Create(Backend backend);
};

// ------------------------------------------------------------------
// Implementation Details (Hidden from standard usage)
// ------------------------------------------------------------------

// 1. MPI Backend
class MpiComm : public Communicator {
    MPI_Comm comm_;

    MPI_Datatype mpiType(DataType t) {
        switch(t) {
            case DataType::FLOAT:  return MPI_FLOAT;
            case DataType::INT:    return MPI_INT;
            case DataType::DOUBLE: return MPI_DOUBLE;
        }
        return MPI_BYTE;
    }

public:
    MpiComm() {
        MPI_Comm_dup(MPI_COMM_WORLD, &comm_); // Duplicate for safety
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }
    ~MpiComm() { MPI_Comm_free(&comm_); }

    void groupStart() override {} // No-op
    void groupEnd() override {}

    struct MpiReq : Request {
        MPI_Request r;
        MpiReq(MPI_Request req) : r(req) {}
        void Wait() override { MPI_Wait(&r, MPI_STATUS_IGNORE); }
    };

    std::unique_ptr<Request> send(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        // Ensure GPU is done computing before MPI reads buffer
        cudaStreamSynchronize(stream); 
        MPI_Request req;
        MPI_Isend(buf, count, mpiType(type), peer, tag, comm_, &req);
        return std::make_unique<MpiReq>(req);
    }

    std::unique_ptr<Request> recv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        // Ensure buffer is free for writing
        cudaStreamSynchronize(stream); 
        MPI_Request req;
        MPI_Irecv(buf, count, mpiType(type), peer, tag, comm_, &req);
        return std::make_unique<MpiReq>(req);
    }
};

// 2. NCCL Backend
class NcclComm : public Communicator {
    ncclComm_t comm_;

    ncclDataType_t ncclType(DataType t) {
        switch(t) {
            case DataType::FLOAT: return ncclFloat;
            case DataType::INT:   return ncclInt;
            case DataType::DOUBLE: return ncclDouble;
        }
        return ncclFloat;
    }

public:
    NcclComm() {
        // Bootstrapping
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        
        ncclUniqueId id;
        if (rank_ == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        ncclCommInitRank(&comm_, size_, id, rank_);
    }
    ~NcclComm() { ncclCommDestroy(comm_); }

    void groupStart() override { ncclGroupStart(); }
    void groupEnd() override   { ncclGroupEnd(); }

    struct NcclReq : Request {
        cudaEvent_t e;
        NcclReq(cudaStream_t s) { cudaEventCreate(&e); cudaEventRecord(e, s); }
        ~NcclReq() { cudaEventDestroy(e); }
        void Wait() override { cudaEventSynchronize(e); }
    };

    std::unique_ptr<Request> send(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        ncclSend(buf, count, ncclType(type), peer, comm_, stream);
        return std::make_unique<NcclReq>(stream);
    }

    std::unique_ptr<Request> recv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        ncclRecv(buf, count, ncclType(type), peer, comm_, stream);
        return std::make_unique<NcclReq>(stream);
    }
};

// ------------------------------------------------------------------
// Factory: The Magic Wrapper
// ------------------------------------------------------------------

std::shared_ptr<Communicator> Communicator::Create(Backend backend) {
    // RAII Static Context: Initialized once, Destroyed at exit
    struct GlobalContext {
        GlobalContext() {
            int init; MPI_Initialized(&init);
            if (!init) MPI_Init(nullptr, nullptr);

            // Auto-detect local rank and set device
            MPI_Comm local_comm;
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
            int local_rank;
            MPI_Comm_rank(local_comm, &local_rank);
            cudaSetDevice(local_rank);
            MPI_Comm_free(&local_comm);
        }
        ~GlobalContext() {
            // Note: In production code, check MPI_Finalized first
            MPI_Finalize(); 
        }
    };

    static GlobalContext ctx; // The singleton that manages lifecycle

    if (backend == Backend::MPI) return std::make_shared<MpiComm>();
    return std::make_shared<NcclComm>();
}

} // namespace
    

#include <vector>
#include <iostream>

#define CUDA_CHECK(call) { if(call != cudaSuccess) exit(1); }

int main() {
    // 1. Library Factory 
    // This automatically: 
    // - Inits MPI
    // - Calculates local rank
    // - Calls cudaSetDevice(local_rank)
    auto comm = GpuComm::Communicator::Create(GpuComm::BackendType::MPI);

    int rank = comm->GetRank();

    // 2. Create Stream & Data (User still owns their data/streams)
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    size_t count = 1024;
    float* d_buffer;
    CUDA_CHECK(cudaMalloc(&d_buffer, count * sizeof(float)));

    // Initialize data
    if (rank == 0) {
        std::vector<float> h_data(count, 1.0f);
        CUDA_CHECK(cudaMemcpy(d_buffer, h_data.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemset(d_buffer, 0, count * sizeof(float)));
    }

    // 3. Portable Communication
    comm->GroupStart(); 

    std::unique_ptr<GpuComm::Request> req;

    if (rank == 0) {
        req = comm->Isend(d_buffer, count, GpuComm::DataType::FLOAT, 1, 0, stream);
    } else if (rank == 1) {
        req = comm->Irecv(d_buffer, count, GpuComm::DataType::FLOAT, 0, 0, stream);
    }

    comm->GroupEnd();

    if (req) {
        req->Wait();
        std::cout << "Rank " << rank << " finished." << std::endl;
    }

    // Cleanup
    cudaFree(d_buffer);
    cudaStreamDestroy(stream);
    
    // Note: No MPI_Finalize() needed here! 
    // The library handles it automatically on exit.
    return 0;
}
  
  


