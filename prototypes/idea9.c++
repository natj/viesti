#include <mpi.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

// ----------------------------------------------------------------------------
// Compile-Time Configuration
// ----------------------------------------------------------------------------

#ifdef HIP_P2P
#include <rccl/rccl.h>
#endif

// New Backend Flag: MPI_ACTIVE_P2P
// Usage: hipcc -DMPI_RMA -DMPI_ACTIVE_P2P gpu_comm_single.cpp -o app

#if !defined(MPI_P2P) && !defined(HIP_P2P) && !defined(MPI_ACTIVE_P2P)
#error "Please define a P2P backend: -DMPI_P2P, -DHIP_P2P, or -DMPI_ACTIVE_P2P"
#endif

// ============================================================================
// LIBRARY DEFINITION
// ============================================================================

namespace GpuComm {

enum class DataType { FLOAT, DOUBLE, INT, UNKNOWN };

template<typename T> struct TypeTraits { static constexpr DataType value = DataType::UNKNOWN; };
template<> struct TypeTraits<float>    { static constexpr DataType value = DataType::FLOAT; };
template<> struct TypeTraits<double>   { static constexpr DataType value = DataType::DOUBLE; };
template<> struct TypeTraits<int>      { static constexpr DataType value = DataType::INT; };

namespace {
    size_t get_type_size(DataType dt) {
        switch(dt) {
            case DataType::FLOAT: return sizeof(float);
            case DataType::DOUBLE: return sizeof(double);
            case DataType::INT: return sizeof(int);
            default: return 1;
        }
    }

    MPI_Datatype to_mpi_type(DataType dt) {
        switch(dt) {
            case DataType::FLOAT: return MPI_FLOAT;
            case DataType::DOUBLE: return MPI_DOUBLE;
            case DataType::INT: return MPI_INT;
            default: return MPI_BYTE;
        }
    }

    void check_hip(hipError_t err, const char* file, int line) {
        if (err != hipSuccess) {
            throw std::runtime_error(std::string("HIP Error: ") + hipGetErrorString(err) + 
                                   " at " + file + ":" + std::to_string(line));
        }
    }
}
#define CHECK_HIP(cmd) check_hip(cmd, __FILE__, __LINE__)

// --------------------------------------------------------------------------
// Concrete Unified Request Class
// --------------------------------------------------------------------------
class Request {
public:
    // New Type: MPI_WIN_WAIT for Active P2P
    enum class Type { MPI_REQ, HIP_EVENT, MPI_WIN_WAIT };

private:
    Type type_;
    bool active_;

    union Handle {
        MPI_Request mpi_req;
        hipEvent_t hip_event;
        MPI_Win mpi_win; // For Active P2P receive wait
    } handle_;

public:
    // Standard MPI Request
    Request(MPI_Request req) : type_(Type::MPI_REQ), active_(true) {
        handle_.mpi_req = req;
    }

    // HIP Event
    Request(hipStream_t stream) : type_(Type::HIP_EVENT), active_(true) {
        CHECK_HIP(hipEventCreate(&handle_.hip_event));
        CHECK_HIP(hipEventRecord(handle_.hip_event, stream));
    }

    // New: Active P2P Window Wait
    // This request represents the receiver waiting for the epoch to close
    Request(MPI_Win win) : type_(Type::MPI_WIN_WAIT), active_(true) {
        handle_.mpi_win = win;
    }
    
    // Default constructor for completed/dummy requests
    Request() : type_(Type::MPI_REQ), active_(false) {
        handle_.mpi_req = MPI_REQUEST_NULL;
    }

    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    
    Request(Request&& other) noexcept : type_(other.type_), active_(other.active_), handle_(other.handle_) {
        other.active_ = false;
    }

    Request& operator=(Request&& other) noexcept {
        if (this != &other) {
            if (active_) wait();
            type_ = other.type_;
            active_ = other.active_;
            handle_ = other.handle_;
            other.active_ = false;
        }
        return *this;
    }

    ~Request() {
        if (active_) wait();
        if (type_ == Type::HIP_EVENT) {
            hipEventDestroy(handle_.hip_event);
        }
    }

    void wait() {
        if (!active_) return;

        if (type_ == Type::MPI_REQ) {
            if (handle_.mpi_req != MPI_REQUEST_NULL) {
                MPI_Wait(&handle_.mpi_req, MPI_STATUS_IGNORE);
            }
        } else if (type_ == Type::HIP_EVENT) {
            CHECK_HIP(hipEventSynchronize(handle_.hip_event));
        } else if (type_ == Type::MPI_WIN_WAIT) {
            // The Receiver calls MPI_Win_wait to finalize the epoch
            // This blocks until the sender has called Complete
            MPI_Win_wait(handle_.mpi_win); 
        }
        active_ = false;
    }
};

// --------------------------------------------------------------------------
// Concrete Communicator Class
// --------------------------------------------------------------------------
class Communicator {
private:
    MPI_Comm mpi_comm_;
    int rank_;
    int size_;
    bool owns_mpi_env_;

    // RMA Resources
    std::map<int, MPI_Win> windows_;
    // Reverse lookup to find Window ID from Pointer (Needed for Active P2P API)
    std::map<const void*, int> ptr_to_id_;

    // Active P2P Optimization: Group Caching
    // Creating groups is expensive, so we cache them: rank -> group
    std::map<int, MPI_Group> group_cache_;
    MPI_Group world_group_;

#ifdef HIP_P2P
    ncclComm_t nccl_comm_;
    hipStream_t stream_;
#endif

    // Helper: Get or Create single-rank group
    MPI_Group get_group_for_rank(int target_rank) {
        if (group_cache_.find(target_rank) == group_cache_.end()) {
            MPI_Group new_group;
            MPI_Group_incl(world_group_, 1, &target_rank, &new_group);
            group_cache_[target_rank] = new_group;
        }
        return group_cache_[target_rank];
    }

public:
    Communicator(int* argc, char*** argv, hipStream_t stream = nullptr) {
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

        // Initialize Group info for Active P2P
        MPI_Comm_group(mpi_comm_, &world_group_);

#ifdef HIP_P2P
        if (!stream) throw std::runtime_error("HIP_P2P requires stream");
        stream_ = stream;
        ncclUniqueId id;
        if (rank_ == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm_);
        ncclCommInitRank(&nccl_comm_, size_, id, rank_);
#endif
    }

    ~Communicator() {
#ifdef HIP_P2P
        ncclCommDestroy(nccl_comm_);
#endif
        // Cleanup Groups
        for(auto& g : group_cache_) MPI_Group_free(&g.second);
        MPI_Group_free(&world_group_);

        for (auto& pair : windows_) MPI_Win_free(&pair.second);
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

    // ========================================================================
    // BUFFER MANAGEMENT (Shared by RMA and ACTIVE_P2P)
    // ========================================================================

    void register_buffer(void* ptr, size_t size_bytes, int buffer_id) {
        if (windows_.count(buffer_id)) throw std::runtime_error("Buffer ID exists");
        MPI_Win win;
        MPI_Win_create(ptr, size_bytes, 1, MPI_INFO_NULL, mpi_comm_, &win);
        windows_[buffer_id] = win;
        ptr_to_id_[ptr] = buffer_id; // Store for Active P2P lookup
    }

    void deregister_buffer(int buffer_id) {
        // Implementation omitted for brevity, logic implies removing from both maps
    }

    // ========================================================================
    // RMA API (Lock/Unlock)
    // ========================================================================

    void lock_buffer(int target_rank, int buffer_id) {
#ifdef MPI_RMA
        MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, windows_.at(buffer_id));
#endif
    }

    void unlock_buffer(int target_rank, int buffer_id) {
#ifdef MPI_RMA
        MPI_Win_unlock(target_rank, windows_.at(buffer_id));
#endif
    }

    Request put(const void* src, size_t count, DataType type, int target_rank, int buffer_id, int tag) {
#ifdef MPI_RMA
        MPI_Request req;
        MPI_Rput(src, count, to_mpi_type(type), target_rank, 0, count, to_mpi_type(type), windows_.at(buffer_id), &req);
        return Request(req);
#else
        throw std::runtime_error("MPI_RMA not enabled");
#endif
    }

    // ========================================================================
    // P2P API (Send/Recv)
    // ========================================================================

    void start_sendrecv() {
#ifdef HIP_P2P
        ncclGroupStart();
#endif
    }

    void end_sendrecv() {
#ifdef HIP_P2P
        ncclGroupEnd();
#endif
    }

    Request send(const void* src, size_t count, DataType type, int target_rank, int tag) {
#ifdef HIP_P2P
        ncclSend(src, count * get_type_size(type), ncclChar, target_rank, nccl_comm_, stream_);
        return Request(stream_);

#elif defined(MPI_P2P)
        MPI_Request req;
        MPI_Isend(src, count, to_mpi_type(type), target_rank, tag, mpi_comm_, &req);
        return Request(req);

#elif defined(MPI_ACTIVE_P2P)
        // --------------------------------------------------------
        // ACTIVE RMA IMPLEMENTATION OF SEND
        // Semantics: Start -> Put -> Complete
        // --------------------------------------------------------
        
        // 1. Look up the window ID associated with the *source* pointer? 
        // NO: In P2P Put, we need the window ID of the TARGET buffer.
        // LIMITATION: Standard send() doesn't know the target's buffer ID.
        // ASSUMPTION: For this backend, we assume src/dest pointers map to 
        // the SAME buffer_id on both ends (Symmetric allocation).
        
        if (ptr_to_id_.find(src) == ptr_to_id_.end()) 
            throw std::runtime_error("Active P2P: Buffer must be registered!");
        
        int bid = ptr_to_id_.at(src);
        MPI_Win win = windows_.at(bid);
        MPI_Group target_group = get_group_for_rank(target_rank);

        // 2. Start Access Epoch
        MPI_Win_start(target_group, 0, win);

        // 3. Put Data
        // We write to offset 0 of the target's registered buffer
        MPI_Put(src, count, to_mpi_type(type), target_rank, 0, count, to_mpi_type(type), win);

        // 4. Complete Epoch
        // This BLOCKS until local buffer is reusable. 
        MPI_Win_complete(win);

        // Since Complete blocks, we return a "Finished" request
        return Request(); 
#endif
    }

    Request recv(void* dest, size_t count, DataType type, int source_rank, int tag) {
#ifdef HIP_P2P
        ncclRecv(dest, count * get_type_size(type), ncclChar, source_rank, nccl_comm_, stream_);
        return Request(stream_);

#elif defined(MPI_P2P)
        MPI_Request req;
        MPI_Irecv(dest, count, to_mpi_type(type), source_rank, tag, mpi_comm_, &req);
        return Request(req);

#elif defined(MPI_ACTIVE_P2P)
        // --------------------------------------------------------
        // ACTIVE RMA IMPLEMENTATION OF RECV
        // Semantics: Post -> Wait
        // --------------------------------------------------------
        
        if (ptr_to_id_.find(dest) == ptr_to_id_.end()) 
            throw std::runtime_error("Active P2P: Dest buffer must be registered!");

        int bid = ptr_to_id_.at(dest);
        MPI_Win win = windows_.at(bid);
        MPI_Group source_group = get_group_for_rank(source_rank);

        // 1. Post: Allow source_rank to write to us
        MPI_Win_post(source_group, 0, win);

        // 2. Return a Request that will call MPI_Win_wait()
        return Request(win);
#endif
    }

    // Templated Helpers
    template <typename T> Request put(const T* src, size_t count, int target_rank, int buffer_id, int tag) {
        return put(src, count, TypeTraits<T>::value, target_rank, buffer_id, tag);
    }
    template <typename T> Request send(const T* src, size_t count, int target_rank, int tag) {
        return send(src, count, TypeTraits<T>::value, target_rank, tag);
    }
    template <typename T> Request recv(T* dest, size_t count, int source_rank, int tag) {
        return recv(dest, count, TypeTraits<T>::value, source_rank, tag);
    }
};

} // namespace GpuComm

// ============================================================================
// MAIN APPLICATION
// ============================================================================

int main(int argc, char** argv) {
  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  // Instantiate
  GpuComm::Communicator comm(&argc, &argv, stream);
  int rank = comm.rank();

  // Init Data
  size_t count = 1024;
  size_t size_bytes = count * sizeof(float);
  float *buf_A, *buf_B;
  
  CHECK_HIP(hipMalloc(&buf_A, size_bytes)); 
  CHECK_HIP(hipMalloc(&buf_B, size_bytes)); 
  
  // NOTE: For MPI_ACTIVE_P2P, buffers MUST be registered first
  comm.register_buffer(buf_B, size_bytes, 20); 
  comm.barrier();

  // Test Active P2P (or standard P2P depending on flag)
  // Logic: Rank 0 sends buf_B to Rank 1's buf_B
  
  std::vector<GpuComm::Request> requests;

  if (rank == 0) {
    // For Active P2P: This calls Start -> Put -> Complete
    // It uses the fact that buf_B is registered to ID 20 to find the window
    requests.push_back(comm.send(buf_B, count, 1, 999));
    std::cout << "Rank 0 Sent." << std::endl;
  } 
  else if (rank == 1) {
    // For Active P2P: This calls Post, returns Request wrapping Wait
    requests.push_back(comm.recv(buf_B, count, 0, 999));
    std::cout << "Rank 1 Posted Recv." << std::endl;
  }

  // Wait for completion
  for(auto& req : requests) req.wait();
  
  if (rank == 1) std::cout << "Rank 1 Finished." << std::endl;

  CHECK_HIP(hipFree(buf_A));
  CHECK_HIP(hipFree(buf_B));
  CHECK_HIP(hipStreamDestroy(stream));
  return 0;
}
