#include "gpu_comm.hpp"
#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <map>
#include <vector>

namespace GpuComm {

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

ncclDataType_t to_nccl_type(DataType dt) {
    switch(dt) {
        case DataType::FLOAT: return ncclFloat;
        case DataType::DOUBLE: return ncclDouble;
        case DataType::INT: return ncclInt;
        default: return ncclChar;
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
// MPI Backend
// --------------------------------------------------------------------------
class MpiRequest : public Request {
    MPI_Request req;
    bool active;
public:
    MpiRequest(MPI_Request r) : req(r), active(true) {}
    MpiRequest() : req(MPI_REQUEST_NULL), active(false) {} // Completed/Dummy request
    
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
        // Passive target synchronization
        MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, windows_.at(buffer_id));
    }

    void unlock_buffer(int target_rank, int buffer_id) override {
        MPI_Win_unlock(target_rank, windows_.at(buffer_id));
    }

    std::unique_ptr<Request> put(const void* src, size_t count, DataType type, int target_rank, int buffer_id, int tag) override {
        // MPI_Rput is Request-based put
        MPI_Request req;
        size_t bytes = count * type_size(type);
        
        // Note: target_disp is 0 (assumed writing to start of buffer based on description)
        MPI_Rput(src, count, to_mpi_type(type), 
                 target_rank, 0, count, to_mpi_type(type), 
                 windows_.at(buffer_id), &req);
        
        // For standard MPI_Put/Get inside lock/unlock, flush might be needed instead of Request
        // But MPI_Rput is valid. We wrap it.
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
// NCCL Backend
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
        // Block host until stream reaches this point
        // Ideally we would use cudaStreamWaitEvent to sync streams, 
        // but the API implies host synchronization.
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

    // --- RMA Not Supported in NCCL ---
    void register_buffer(void* ptr, size_t size_bytes, int buffer_id) override {} 
    void deregister_buffer(int buffer_id) override {}
    void lock_buffer(int target_rank, int buffer_id) override {}
    void unlock_buffer(int target_rank, int buffer_id) override {}

    std::unique_ptr<Request> put(const void* src, size_t count, DataType type, int target_rank, int buffer_id, int tag) override {
        throw std::runtime_error("NCCL Backend does not support One-sided Put");
    }

    // --- P2P ---
    void start_sendrecv() override {
        ncclGroupStart();
    }

    void end_sendrecv() override {
        ncclGroupEnd();
    }

    std::unique_ptr<Request> send(const void* src, size_t count, DataType type, int target_rank, int tag) override {
        // Note: Tag is ignored in NCCL send/recv usually, relies on ordering
        ncclSend(src, count * type_size(type), ncclChar, target_rank, nccl_comm_, stream_);
        return std::make_unique<NcclRequest>(stream_);
    }

    std::unique_ptr<Request> recv(void* dest, size_t count, DataType type, int source_rank, int tag) override {
        ncclRecv(dest, count * type_size(type), ncclChar, source_rank, nccl_comm_, stream_);
        return std::make_unique<NcclRequest>(stream_);
    }
};

// --------------------------------------------------------------------------
// Factory
// --------------------------------------------------------------------------
std::unique_ptr<Communicator> Communicator::create(Backend backend, MPI_Comm mpi_comm, cudaStream_t stream) {
    if (backend == Backend::MPI) {
        return std::make_unique<MpiCommunicator>(mpi_comm);
    } else {
        return std::make_unique<NcclCommunicator>(mpi_comm, stream);
    }
}

} // namespace
