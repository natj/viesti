#pragma once

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <map>
#include <iostream>
#include <algorithm>

namespace GpuComm {

// ------------------------------------------------------------------
// Enums & Types
// ------------------------------------------------------------------
enum class Backend  { MPI, NCCL, MPI_RMA };
enum class DataType { FLOAT, INT, DOUBLE };

// ------------------------------------------------------------------
// Abstract Interfaces
// ------------------------------------------------------------------

// A handle to track asynchronous operations
struct Request {
    virtual ~Request() = default;
    // Blocks CPU until operation completes
    virtual void Wait() = 0; 
};

class Communicator {
protected:
    int rank_ = 0;
    int size_ = 1;

public:
    virtual ~Communicator() = default;
    
    // Getters
    int rank() const { return rank_; }
    int size() const { return size_; }

    // --- Registration (Required for RMA) ---
    // 'tag' acts as the Identifier for this buffer during communication
    virtual void registerMemory(void* ptr, size_t size, int tag) = 0;

    // --- Synchronization ---
    virtual void groupStart() = 0;
    virtual void groupEnd() = 0;

    // --- P2P Communication ---
    virtual std::unique_ptr<Request> send(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) = 0;
    virtual std::unique_ptr<Request> recv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) = 0;

    // --- Factory ---
    static std::shared_ptr<Communicator> Create(Backend backend);
};

// ------------------------------------------------------------------
// 1. Two-Sided MPI Backend (Standard)
// ------------------------------------------------------------------
class MpiComm : public Communicator {
private:
    MPI_Comm comm_;

    MPI_Datatype mpiType(DataType t) {
        if (t == DataType::FLOAT) return MPI_FLOAT;
        if (t == DataType::INT) return MPI_INT;
        return MPI_DOUBLE;
    }

    struct MpiReq : Request {
        MPI_Request r;
        MpiReq(MPI_Request req) : r(req) {}
        void Wait() override { MPI_Wait(&r, MPI_STATUS_IGNORE); }
    };

public:
    MpiComm() {
        MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }
    
    ~MpiComm() override { MPI_Comm_free(&comm_); }

    // Registration is not needed for standard Isend/Irecv
    void registerMemory(void* ptr, size_t size, int tag) override {} 

    void groupStart() override {} // No-op
    void groupEnd() override {}   // No-op

    std::unique_ptr<Request> send(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        // Ensure GPU is done computing before MPI reads
        cudaStreamSynchronize(stream); 
        MPI_Request req;
        MPI_Isend(buf, count, mpiType(type), peer, tag, comm_, &req);
        return std::make_unique<MpiReq>(req);
    }

    std::unique_ptr<Request> recv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        // Ensure buffer is free to write
        cudaStreamSynchronize(stream); 
        MPI_Request req;
        MPI_Irecv(buf, count, mpiType(type), peer, tag, comm_, &req);
        return std::make_unique<MpiReq>(req);
    }
};

// ------------------------------------------------------------------
// 2. NCCL Backend
// ------------------------------------------------------------------
class NcclComm : public Communicator {
private:
    ncclComm_t comm_;

    ncclDataType_t ncclType(DataType t) {
        if (t == DataType::FLOAT) return ncclFloat;
        if (t == DataType::INT) return ncclInt;
        return ncclDouble;
    }

    struct NcclReq : Request {
        cudaEvent_t e;
        NcclReq(cudaStream_t s) { cudaEventCreate(&e); cudaEventRecord(e, s); }
        ~NcclReq() { cudaEventDestroy(e); }
        void Wait() override { cudaEventSynchronize(e); }
    };

public:
    NcclComm() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        
        ncclUniqueId id;
        if (rank_ == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // Assumes cudaSetDevice was already handled by the Factory
        ncclCommInitRank(&comm_, size_, id, rank_);
    }

    ~NcclComm() override { ncclCommDestroy(comm_); }

    // NCCL handles memory pointers dynamically, no registration needed
    void registerMemory(void* ptr, size_t size, int tag) override {} 

    void groupStart() override { ncclGroupStart(); }
    void groupEnd() override   { ncclGroupEnd(); }

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
// 3. One-Sided MPI (RMA) Backend
// ------------------------------------------------------------------
class MpiRmaComm : public Communicator {
private:
    MPI_Win win_;
    
    // Maps Tag -> Vector of Addresses (Index = Rank)
    std::map<int, std::vector<MPI_Aint>> address_registry_;

    MPI_Datatype mpiType(DataType t) {
        if (t == DataType::FLOAT) return MPI_FLOAT;
        if (t == DataType::INT) return MPI_INT;
        return MPI_DOUBLE;
    }

    struct NoOpReq : Request {
        void Wait() override {} // Waiting is handled by groupEnd (Fence)
    };

public:
    MpiRmaComm() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        
        // Create a Dynamic Window that allows attaching memory later
        MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &win_);
    }

    ~MpiRmaComm() override {
        MPI_Win_free(&win_);
    }

    // Stores the pointer and exchanges addresses globally
    void registerMemory(void* ptr, size_t size, int tag) override {
        // 1. Authorize MPI to access this memory
        MPI_Win_attach(win_, ptr, size);

        // 2. Get local absolute address
        MPI_Aint local_addr;
        MPI_Get_address(ptr, &local_addr);

        // 3. Prepare storage
        address_registry_[tag].resize(size_);

        // 4. Exchange addresses
        MPI_Allgather(&local_addr, 1, MPI_AINT, 
                      address_registry_[tag].data(), 1, MPI_AINT, 
                      MPI_COMM_WORLD);
    }

    // Synchronization Epochs (Fence)
    void groupStart() override { MPI_Win_fence(0, win_); }
    void groupEnd() override   { MPI_Win_fence(0, win_); }

    std::unique_ptr<Request> send(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        cudaStreamSynchronize(stream); // Sync before MPI reads

        // 1. Find the target address based on the Tag
        auto it = address_registry_.find(tag);
        if (it == address_registry_.end()) {
            std::cerr << "[MpiRma] Error: Tag " << tag << " not registered!" << std::endl;
            std::terminate();
        }

        MPI_Aint target_disp = it->second[peer];

        // 2. Perform Put
        MPI_Put(buf, count, mpiType(type), 
                peer, target_disp, count, mpiType(type), 
                win_);
        
        return std::make_unique<NoOpReq>();
    }

    std::unique_ptr<Request> recv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        // Passive receive. Data will arrive before groupEnd returns.
        return std::make_unique<NoOpReq>();
    }
};

// ------------------------------------------------------------------
// Factory Implementation (Streamlined Initialization)
// ------------------------------------------------------------------
std::shared_ptr<Communicator> Communicator::Create(Backend backend) {
    // Singleton context to handle Init/Finalize exactly once
    struct GlobalContext {
        GlobalContext() {
            int init; MPI_Initialized(&init);
            if (!init) MPI_Init(nullptr, nullptr);

            // Auto-detect local rank and set device
            MPI_Comm local_comm;
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
            int local_rank;
            MPI_Comm_rank(local_comm, &local_rank);
            
            cudaError_t err = cudaSetDevice(local_rank);
            if (err != cudaSuccess) {
                std::cerr << "Failed to set device: " << local_rank << std::endl;
            }

            MPI_Comm_free(&local_comm);
        }
        ~GlobalContext() {
            // Check Finalized to prevent double-free issues in some edge cases
            int fin; MPI_Finalized(&fin);
            if (!fin) MPI_Finalize(); 
        }
    };

    static GlobalContext ctx; 

    switch(backend) {
        case Backend::MPI:     return std::make_shared<MpiComm>();
        case Backend::NCCL:    return std::make_shared<NcclComm>();
        case Backend::MPI_RMA: return std::make_shared<MpiRmaComm>();
    }
    return nullptr;
}

} // namespace GpuComm



int main() {

    //auto comm = GpuComm::Communicator::Create(GpuComm::Backend::NCCL);
    //auto comm = GpuComm::Communicator::Create(GpuComm::Backend::MPI);
    auto comm = GpuComm::Communicator::Create(GpuComm::Backend::MPI_RMA);
    int rank = comm->rank();

    cudaStream_t stream; 
    cudaStreamCreate(&stream);

    // --- Allocate Two Buffers ---
    size_t size = 1024 * sizeof(float);
    float *buf_A, *buf_B;
    cudaMalloc(&buf_A, size);
    cudaMalloc(&buf_B, size);

    // --- Register Both with distinct Tags ---
    // Tag 10 = Buffer A
    comm->registerMemory(buf_A, size, 10);
    // Tag 20 = Buffer B
    comm->registerMemory(buf_B, size, 20);

    //--------------------------------------------------
    comm->groupStart();

    if (rank == 0) {
        // Send 1s to Rank 1's Buffer A (Tag 10)
        // Note: The 'send' buffer can be anything, but target is determined by Tag
        comm->send(buf_A, 1024, GpuComm::DataType::FLOAT, 1, 10, stream);

        // Send 2s to Rank 1's Buffer B (Tag 20)
        comm->send(buf_A, 1024, GpuComm::DataType::FLOAT, 1, 20, stream);
    } 
    else if (rank == 1) {
        // Recv calls are passive markers here, but good for symmetry
        comm->recv(buf_A, 1024, GpuComm::DataType::FLOAT, 0, 10, stream);
        comm->recv(buf_B, 1024, GpuComm::DataType::FLOAT, 0, 20, stream);
    }

    comm->groupEnd();

    // wait until finished
    if (req) {
        req->Wait();
        std::cout << "Rank " << rank << " finished." << std::endl;
    }

    // Cleanup
    cudaFree(d_buffer);
    cudaStreamDestroy(stream);
    
    return 0;
}
