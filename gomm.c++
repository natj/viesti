#include <mpi.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

// ----------------------------------------------------------------------------
// Compile-Time Configuration Checks
// ----------------------------------------------------------------------------

// Check if RCCL headers are needed
#ifdef HIP_P2P
#include <rccl/rccl.h>
#endif

// Ensure at least one P2P backend is selected
#if !defined(MPI_P2P) && !defined(HIP_P2P)
#error "Please define at least one P2P backend: -DMPI_P2P or -DHIP_P2P"
#endif

// Ensure at least one RMA backend is selected
#if !defined(MPI_RMA) && !defined(HIP_RMA)
#error "Please define at least one RMA backend: -DMPI_RMA or -DHIP_RMA"
#endif

// ============================================================================
// LIBRARY DEFINITION
// ============================================================================

namespace GpuComm {

enum class DataType { FLOAT, DOUBLE, INT, UNKNOWN };

// Type Traits for Automatic Type Deduction
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
#define CHECK_HIP(cmd) GpuComm::check_hip(cmd, __FILE__, __LINE__)

// --------------------------------------------------------------------------
// Concrete Unified Request Class
// --------------------------------------------------------------------------
class Request {
public:
    enum class Type { MPI, HIP };

private:
    Type type_;
    bool active_;

    // Tagged Union to hold backend-specific handles
    union Handle {
        MPI_Request mpi_req;
        hipEvent_t hip_event;
    } handle_;

public:
    // MPI Constructor
    Request(MPI_Request req) : type_(Type::MPI), active_(true) {
        handle_.mpi_req = req;
    }

    // HIP Constructor
    Request(hipStream_t stream) : type_(Type::HIP), active_(true) {
        CHECK_HIP(hipEventCreate(&handle_.hip_event));
        CHECK_HIP(hipEventRecord(handle_.hip_event, stream));
    }

    // Move Semantics Only (RAII)
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
        if (type_ == Type::HIP) {
            // Note: In production, ensure device context is still valid
            hipEventDestroy(handle_.hip_event);
        }
    }

    void wait() {
        if (!active_) return;

        if (type_ == Type::MPI) {
            if (handle_.mpi_req != MPI_REQUEST_NULL) {
                MPI_Wait(&handle_.mpi_req, MPI_STATUS_IGNORE);
            }
        } else {
            CHECK_HIP(hipEventSynchronize(handle_.hip_event));
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

    // RMA Resources (MPI Backend)
#ifdef MPI_RMA
    std::map<int, MPI_Win> windows_;
#endif

    // P2P Resources (HIP/RCCL Backend)
#ifdef HIP_P2P
    ncclComm_t nccl_comm_;
    hipStream_t stream_;
#endif

public:
    Communicator(int* argc, char*** argv, hipStream_t stream = nullptr) {
        // 1. Initialize MPI (Required for bootstrap even in full HIP mode)
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

        // 2. Initialize RCCL if HIP_P2P is enabled
#ifdef HIP_P2P
        if (!stream) throw std::runtime_error("HIP_P2P backend requires a valid stream");
        stream_ = stream;
        
        ncclUniqueId id;
        if (rank_ == 0) ncclGetUniqueId(&id);
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm_);
        
        ncclResult_t res = ncclCommInitRank(&nccl_comm_, size_, id, rank_);
        if (res != ncclSuccess) throw std::runtime_error("RCCL Init failed");
#endif
    }

    ~Communicator() {
#ifdef HIP_P2P
        ncclCommDestroy(nccl_comm_);
#endif

#ifdef MPI_RMA
        for (auto& pair : windows_) {
            MPI_Win_free(&pair.second);
        }
#endif
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
    // ONE-SIDED (RMA) API
    // ========================================================================

    void register_buffer(void* ptr, size_t size_bytes, int buffer_id) {
#ifdef MPI_RMA
        if (windows_.count(buffer_id)) throw std::runtime_error("Buffer ID exists");
        MPI_Win win;
        MPI_Win_create(ptr, size_bytes, 1, MPI_INFO_NULL, mpi_comm_, &win);
        windows_[buffer_id] = win;
#elif defined(HIP_RMA)
        // Not implemented (Placeholder)
#endif
    }

    void deregister_buffer(int buffer_id) {
#ifdef MPI_RMA
        auto it = windows_.find(buffer_id);
        if (it != windows_.end()) {
            MPI_Win_free(&it->second);
            windows_.erase(it);
        }
#elif defined(HIP_RMA)
        // Not implemented
#endif
    }

    void lock_buffer(int target_rank, int buffer_id) {
#ifdef MPI_RMA
        MPI_Win_lock(MPI_LOCK_SHARED, target_rank, 0, windows_.at(buffer_id));
#elif defined(HIP_RMA)
        // Not implemented
#endif
    }

    void unlock_buffer(int target_rank, int buffer_id) {
#ifdef MPI_RMA
        MPI_Win_unlock(target_rank, windows_.at(buffer_id));
#elif defined(HIP_RMA)
        // Not implemented
#endif
    }

    // TODO: methods to lock/unlock all buffers


    Request put(const void* src, size_t count, DataType type, int target_rank, int buffer_id) {
#ifdef MPI_RMA
        MPI_Request req;
        MPI_Rput(src, count, to_mpi_type(type), target_rank, 0, count, to_mpi_type(type), windows_.at(buffer_id), &req);
        return Request(req);
#elif defined(HIP_RMA)
        throw std::runtime_error("HIP_RMA Put not implemented");
#else
        throw std::runtime_error("No RMA backend configured");
#endif
    }

    // ========================================================================
    // TWO-SIDED (P2P) API
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
#endif
    }

    // ========================================================================
    // Templated Helpers
    // ========================================================================
    template <typename T>
    Request put(const T* src, size_t count, int target_rank, int buffer_id, int tag) {
        return put(src, count, TypeTraits<T>::value, target_rank, buffer_id, tag);
    }

    template <typename T>
    Request send(const T* src, size_t count, int target_rank, int tag) {
        return send(src, count, TypeTraits<T>::value, target_rank, tag);
    }

    template <typename T>
    Request recv(T* dest, size_t count, int source_rank, int tag) {
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

  // --- Display Active Configuration ---
  std::cout << "GpuComm Configuration:" << std::endl;
#ifdef MPI_RMA
  std::cout << "  RMA Backend: MPI" << std::endl;
#elif defined(HIP_RMA)
  std::cout << "  RMA Backend: HIP (Not Implemented)" << std::endl;
#endif

#ifdef MPI_P2P
  std::cout << "  P2P Backend: MPI" << std::endl;
#elif defined(HIP_P2P)
  std::cout << "  P2P Backend: HIP (RCCL)" << std::endl;
#endif


  // Instantiate directly
  GpuComm::Communicator comm(&argc, &argv, stream);
  int rank = comm.rank();

  // --- Data Init ---
  size_t count = 1024;
  size_t size_bytes = count * sizeof(float);
  float *buf_A, *buf_B, *buf_C;
  
  CHECK_HIP(hipMalloc(&buf_A, size_bytes)); 
  CHECK_HIP(hipMalloc(&buf_B, size_bytes)); 
  CHECK_HIP(hipMalloc(&buf_C, size_bytes)); 

  std::vector<float> h_A(count, rank == 0 ? 1.0f : 0.0f);
  std::vector<float> h_B(count, rank == 0 ? 2.0f : 0.0f);
  std::vector<float> h_C(count, rank == 0 ? 3.0f : 0.0f);
  
  CHECK_HIP(hipMemcpy(buf_A, h_A.data(), size_bytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(buf_B, h_B.data(), size_bytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(buf_C, h_C.data(), size_bytes, hipMemcpyHostToDevice));


  // --------------------------------------------------
  // TEST 1: ONE SIDED COMMUNICATION
  // --------------------------------------------------
  int buf_id_A = 10;
  int buf_id_B = 20;
  comm.register_buffer(buf_A, size_bytes, buf_id_A);
  comm.register_buffer(buf_B, size_bytes, buf_id_B);
  comm.barrier();
  
  int target_rank = 1; 
  comm.lock_buffer(target_rank, buf_id_A);
  comm.lock_buffer(target_rank, buf_id_B);

  std::vector<GpuComm::Request> reqs_rma;

  if (rank == 0) {
    reqs_rma.push_back( comm.put(buf_A, count, target_rank, buf_id_A) );
    reqs_rma.push_back( comm.put(buf_B, count, target_rank, buf_id_B) );

    for(auto& req : reqs) req.wait();
    std::cout << rank << ": rank 0 Put data (RMA) into A and B buffers." << std::endl;
  }
  
  comm.unlock_buffer(target_rank, buf_id_A);
  comm.unlock_buffer(target_rank, buf_id_B);
  comm.barrier();


  // --------------------------------------------------
  // TEST 2: SEND / RECV
  // --------------------------------------------------
  comm.start_sendrecv();

  std::vector<GpuComm::Request> reqs_p2p;
  int tag = 42;

  if (rank == 0) {
    reqs_p2p.push_back(comm.send(buf_C, count, 1, tag));
  } else if (rank == 1) {
    reqs_p2p.push_back(comm.recv(buf_C, count, 0, tag));
  }

  comm.end_sendrecv();

  for(auto& req : reqs_p2p) req.wait();
  
  std::cout << rank << ": finished send/recv into C buffer." << std::endl;



  //--------------------------------------------------
  CHECK_HIP(hipFree(buf_A));
  CHECK_HIP(hipFree(buf_B));
  CHECK_HIP(hipFree(buf_C));
  CHECK_HIP(hipStreamDestroy(stream));

  return 0;
}
