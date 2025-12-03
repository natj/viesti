#pragma once

#include <mpi.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <unordered_map> 
#include <iostream>
#include <algorithm>

namespace GpuComm {

enum class Backend { MPI, NCCL, MPI_RMA }; 
enum class DataType { FLOAT, INT, DOUBLE };

// Abstract Interfaces
  
struct Request {
    virtual ~Request() = default;
    virtual void wait() = 0; 
};

class Communicator {
protected:
    int rank_ = 0;
    int size_ = 1;

public:
    virtual ~Communicator() = default;
    
    int rank() const { return rank_; }
    int size() const { return size_; }

    virtual void register_buffer(void* ptr, size_t size, int tag) = 0;

    virtual void comm_start() = 0;
    virtual void comm_end() = 0;

    virtual std::unique_ptr<Request> send(const void* src_buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) = 0;
    virtual std::unique_ptr<Request> recv(void* dst_buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) = 0;

    static std::shared_ptr<Communicator> create(Backend backend);
};


// MPI Two-Sided (Standard)
class MpiTwoSidedComm : public Communicator {
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
        void wait() override { MPI_Wait(&r, MPI_STATUS_IGNORE); }
    };

public:
    MpiTwoSidedComm() {
        MPI_Comm_dup(MPI_COMM_WORLD, &comm_);
        MPI_Comm_rank(comm_, &rank_);
        MPI_Comm_size(comm_, &size_);
    }
    ~MpiTwoSidedComm() override { MPI_Comm_free(&comm_); }

    void register_buffer(void*, size_t, int) override {} // no-op
    void comm_start() override {} // no-op
    void comm_end() override {} // no-op

    std::unique_ptr<Request> send(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        cudaStreamSynchronize(stream); 
        MPI_Request req;
        MPI_Isend(buf, count, mpiType(type), peer, tag, comm_, &req);
        return std::make_unique<MpiReq>(req);
    }

    std::unique_ptr<Request> recv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        cudaStreamSynchronize(stream); 
        MPI_Request req;
        MPI_Irecv(buf, count, mpiType(type), peer, tag, comm_, &req);
        return std::make_unique<MpiReq>(req);
    }
};


// NCCL Backend
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
        void wait() override { cudaEventSynchronize(e); }
    };

public:
    NcclComm() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        ncclUniqueId id;
        if (rank_ == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclCommInitRank(&comm_, size_, id, rank_);
    }
    ~NcclComm() override { ncclCommDestroy(comm_); }

    void register_buffer(void*, size_t, int) override {} 
    void comm_start() override { ncclGroupStart(); }
    void comm_end() override   { ncclGroupEnd(); } // this is when the job is submitted to GPU

    std::unique_ptr<Request> send(const void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        ncclSend(buf, count, ncclType(type), peer, comm_, stream);
        return std::make_unique<NcclReq>(stream);
    }

    std::unique_ptr<Request> recv(void* buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        ncclRecv(buf, count, ncclType(type), peer, comm_, stream);
        return std::make_unique<NcclReq>(stream);
    }
};


// MPI One-Sided (RMA)
class MpiOneSidedComm : public Communicator {
private:
    MPI_Win win_;
    std::unordered_map<int, std::vector<MPI_Aint>> registry_;

    MPI_Datatype mpiType(DataType t) {
        if (t == DataType::FLOAT) return MPI_FLOAT;
        if (t == DataType::INT) return MPI_INT;
        return MPI_DOUBLE;
    }

    struct NoOpReq : Request {
        void wait() override {} // Waiting happens in comm_end
    };

public:
    MpiOneSidedComm() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        MPI_Win_create_dynamic(MPI_INFO_NULL, MPI_COMM_WORLD, &win_);
    }

    ~MpiOneSidedComm() override { MPI_Win_free(&win_); }

    void register_buffer(void* ptr, size_t size, int tag) override {
        MPI_Win_attach(win_, ptr, size);
        MPI_Aint local_addr;
        MPI_Get_address(ptr, &local_addr);
        
        registry_[tag].resize(size_);
        MPI_Allgather(&local_addr, 1, MPI_AINT, 
                      registry_[tag].data(), 1, MPI_AINT, 
                      MPI_COMM_WORLD);
    }

    void comm_start() override { MPI_Win_fence(0, win_); }
    void comm_end() override   { MPI_Win_fence(0, win_); }

    std::unique_ptr<Request> send(const void* src_buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        cudaStreamSynchronize(stream); 

        // buffer lookup
        auto it = registry_.find(tag);
        if (it == registry_.end()) {
            std::cerr << "Error: Unregistered tag " << tag << std::endl;
            std::terminate();
        }

        MPI_Aint target_disp = it->second[peer];
        MPI_Put(src_buf, count, mpiType(type), peer, target_disp, count, mpiType(type), win_);
        
        return std::make_unique<NoOpReq>();
    }

    std::unique_ptr<Request> recv(void* dst_buf, size_t count, DataType type, int peer, int tag, cudaStream_t stream) override {
        return std::make_unique<NoOpReq>();
    }
};


// Factory
std::shared_ptr<Communicator> Communicator::create(Backend backend) {
    struct GlobalContext {
        GlobalContext() {

            // MPI init
            int init; 
            MPI_Initialized(&init);
            if (!init) MPI_Init(nullptr, nullptr);

            // communicator setup
            MPI_Comm local;
            MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);

            int lr; 
            MPI_Comm_rank(local, &lr);
            cudaSetDevice(lr);
            MPI_Comm_free(&local);
        }
        ~GlobalContext() { 
            int fin; 
            MPI_Finalized(&fin); 
            if(!fin) MPI_Finalize(); 
        }
    };
    static GlobalContext ctx; 

    switch(backend) {
        case Backend::MPI:     return std::make_shared<MpiTwoSidedComm>();
        case Backend::NCCL:    return std::make_shared<NcclComm>();
        case Backend::MPI_RMA: return std::make_shared<MpiOneSidedComm>();
    }
    return nullptr;
}

} // namespace GpuComm



int main() {
    auto comm = GpuComm::Communicator::create(GpuComm::Backend::MPI);
    //auto comm = GpuComm::Communicator::create(GpuComm::Backend::MPI_RMA);
    //auto comm = GpuComm::Communicator::create(GpuComm::Backend::NCCL);
    int rank = comm->rank();

    cudaStream_t stream; 
    cudaStreamCreate(&stream);

    size_t count = 1024;
    size_t size_bytes = count * sizeof(float);
    float *buf_A, *buf_B;
    cudaMalloc(&buf_A, size_bytes);
    cudaMalloc(&buf_B, size_bytes);

    // Initialize data 
    if (rank == 0) {
        std::vector<float> h_A(count, 1.0f); // A = 1.0f
        std::vector<float> h_B(count, 2.0f); // B = 2.0f
        cudaMemcpy(buf_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(buf_B, h_B.data(), size_bytes, cudaMemcpyHostToDevice);
    } else {
        cudaMemset(buf_A, 0, size_bytes);
        cudaMemset(buf_B, 0, size_bytes);
    }

    // Expose buffers with tags
    comm->register_buffer(buf_A, size_bytes, 10);
    comm->register_buffer(buf_B, size_bytes, 20);

    //--------------------------------------------------
    // communication epoch starts here
    comm->comm_start();

    // Collection of requests to wait on
    std::vector<std::unique_ptr<GpuComm::Request>> requests;

    if (rank == 0) {
        requests.push_back(comm->send(buf_A, count, GpuComm::DataType::FLOAT, 1, 10, stream));
        requests.push_back(comm->send(buf_B, count, GpuComm::DataType::FLOAT, 1, 20, stream)); 
    } 
    else if (rank == 1) {
        requests.push_back(comm->recv(buf_A, count, GpuComm::DataType::FLOAT, 0, 10, stream));
        requests.push_back(comm->recv(buf_B, count, GpuComm::DataType::FLOAT, 0, 20, stream));
    }

    //--------------------------------------------------
    // communication epoch ends here
    comm->comm_end(); 

    // Wait for all requests (semantics depend on backend)
    for(auto& req : requests) {
        req->wait();
    }

    if (rank == 1) {
        std::cout << "Rank 1 finished receiving." << std::endl;
    }

    // Cleanup
    cudaFree(buf_A);
    cudaFree(buf_B);
    cudaStreamDestroy(stream);
    
    return 0;
}
