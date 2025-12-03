#pragma once

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <iostream>
#include <mutex>
#include <cstdlib> // for std::atexit

namespace GpuComm {

// ------------------------------------------------------------------
// Enums & Types
// ------------------------------------------------------------------
enum class BackendType { MPI, NCCL };
enum class DataType { FLOAT, INT, DOUBLE };

// ------------------------------------------------------------------
// Internal: Global Environment Manager (Singleton)
// ------------------------------------------------------------------
// This handles the "Streamlined" initialization and cleanup
class Environment {
public:
    static void Initialize() {
        static std::once_flag flag;
        std::call_once(flag, []() {
            // 1. MPI Init
            int initialized;
            MPI_Initialized(&initialized);
            if (!initialized) {
                // We assume ownership of Init/Finalize
                MPI_Init(nullptr, nullptr);
                std::atexit([](){ MPI_Finalize(); });
            }

            // 2. Automatic Device Selection
            // We need the Local Rank (rank on this specific node) to pick the GPU.
            MPI_Comm local_comm;
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
            
            int local_rank;
            MPI_Comm_rank(local_comm, &local_rank);
            
            // Bind process to GPU
            cudaError_t err = cudaSetDevice(local_rank);
            if (err != cudaSuccess) {
                std::cerr << "Failed to set device to " << local_rank << std::endl;
                std::terminate();
            }

            MPI_Comm_free(&local_comm);
        });
    }
};

// ------------------------------------------------------------------
// Request Interface
// ------------------------------------------------------------------
class Request {
public:
    virtual ~Request() = default;
    virtual void Wait() = 0;
    virtual bool Test() = 0;
};

// ------------------------------------------------------------------
// Communicator Interface
// ------------------------------------------------------------------
class Communicator {
protected:
    int rank_;
    int size_;

public:
    virtual ~Communicator() = default;

    virtual int GetRank() const { return rank_; }
    virtual int GetSize() const { return size_; }

    virtual void GroupStart() = 0;
    virtual void GroupEnd() = 0;

    virtual std::unique_ptr<Request> Isend(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) = 0;
    virtual std::unique_ptr<Request> Irecv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) = 0;

    // --- Factory ---
    static std::shared_ptr<Communicator> Create(BackendType backend);
};

// ------------------------------------------------------------------
// Concrete: MPI Implementation
// ------------------------------------------------------------------
class MpiCommunicator : public Communicator {
private:
    MPI_Comm comm_;
    MPI_Datatype GetMpiType(DataType t) {
        // Simple mapping for the sketch
        if(t == DataType::FLOAT) return MPI_FLOAT;
        return MPI_BYTE; 
    }

public:
    MpiCommunicator() {
        // Environment::Initialize() is guaranteed to have run by the Factory
        MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }
    
    ~MpiCommunicator() override {
        MPI_Comm_free(&comm_);
    }

    void GroupStart() override {} // No-op
    void GroupEnd() override {}   // No-op

    // Request Wrapper for MPI
    class MpiRequest : public Request {
        MPI_Request req_;
    public:
        MpiRequest(MPI_Request r) : req_(r) {}
        void Wait() override { MPI_Wait(&req_, MPI_STATUS_IGNORE); }
        bool Test() override { int flag; MPI_Test(&req_, &flag, MPI_STATUS_IGNORE); return flag; }
    };

    std::unique_ptr<Request> Isend(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        // Ideally: cudaStreamSynchronize(stream) here if MPI is not CUDA-aware
        MPI_Request req;
        MPI_Isend(buf, count, GetMpiType(type), peer, tag, comm_, &req);
        return std::make_unique<MpiRequest>(req);
    }

    std::unique_ptr<Request> Irecv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        MPI_Request req;
        MPI_Irecv(buf, count, GetMpiType(type), peer, tag, comm_, &req);
        return std::make_unique<MpiRequest>(req);
    }
};

// ------------------------------------------------------------------
// Concrete: NCCL Implementation
// ------------------------------------------------------------------
class NcclCommunicator : public Communicator {
private:
    ncclComm_t nccl_comm_;
    
    ncclDataType_t GetNcclType(DataType t) {
        if(t == DataType::FLOAT) return ncclFloat;
        return ncclChar;
    }

public:
    NcclCommunicator() {
        // 1. Get info from MPI (Bootstrapping)
        int mpi_rank, mpi_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
        
        this->rank_ = mpi_rank;
        this->size_ = mpi_size;

        // 2. Exchange Unique ID
        ncclUniqueId id;
        if (rank_ == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

        // 3. Init NCCL
        // Note: cudaSetDevice was already called by Environment::Initialize
        ncclCommInitRank(&nccl_comm_, size_, id, rank_);
    }

    ~NcclCommunicator() override {
        ncclCommDestroy(nccl_comm_);
    }

    void GroupStart() override { ncclGroupStart(); }
    void GroupEnd() override { ncclGroupEnd(); }

    // Request Wrapper for NCCL
    class NcclRequest : public Request {
        cudaEvent_t event_;
    public:
        NcclRequest(cudaStream_t stream) {
            cudaEventCreate(&event_);
            cudaEventRecord(event_, stream);
        }
        ~NcclRequest() { cudaEventDestroy(event_); }
        void Wait() override { cudaEventSynchronize(event_); }
        bool Test() override { return cudaEventQuery(event_) == cudaSuccess; }
    };

    std::unique_ptr<Request> Isend(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        ncclSend(buf, count, GetNcclType(type), peer, nccl_comm_, stream);
        return std::make_unique<NcclRequest>(stream);
    }

    std::unique_ptr<Request> Irecv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        ncclRecv(buf, count, GetNcclType(type), peer, nccl_comm_, stream);
        return std::make_unique<NcclRequest>(stream);G
    }
};

// ------------------------------------------------------------------
// Factory Implementation
// ------------------------------------------------------------------
std::shared_ptr<Communicator> Communicator::Create(BackendType backend) {
    // Hidden step: Ensure MPI Init and cudaSetDevice are done
    Environment::Initialize();

    if (backend == BackendType::MPI) {
        return std::make_shared<MpiCommunicator>();
    } else {
        return std::make_shared<NcclCommunicator>();
    }
}

} // namespace GpuComm
  

#include "GpuComm.hpp"
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
  
  
