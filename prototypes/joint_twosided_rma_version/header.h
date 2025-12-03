#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <map>

// Forward declarations for backend specific types
typedef struct ompi_communicator_t* MPI_Comm;
typedef struct CUstream_st* cudaStream_t;

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
    // This must be collective (all ranks call it with the same ID)
    virtual void register_buffer(void* ptr, size_t size_bytes, int buffer_id) = 0;
    
    // Unregisters buffer
    virtual void deregister_buffer(int buffer_id) = 0;

    // Locks a specific remote buffer for access
    virtual void lock_buffer(int target_rank, int buffer_id) = 0;
    
    // Unlocks
    virtual void unlock_buffer(int target_rank, int buffer_id) = 0;

    // Put data from local 'src' to remote 'buffer_id' on 'target_rank'
    // Returns a request mainly for tracking local completion
    virtual std::unique_ptr<Request> put(const void* src, size_t count, DataType type, int target_rank, int buffer_id, int tag) = 0;

    // --- Two-Sided (Send/Recv) ---
    virtual void start_sendrecv() = 0; // Group start
    virtual void end_sendrecv() = 0;   // Group end

    virtual std::unique_ptr<Request> send(const void* src, size_t count, DataType type, int target_rank, int tag) = 0;
    virtual std::unique_ptr<Request> recv(void* dest, size_t count, DataType type, int source_rank, int tag) = 0;
};

} // namespace GpuComm
