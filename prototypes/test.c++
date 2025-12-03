// Common setup code for both test cases
#include "GpuComm.hpp" // The header we designed
#include <vector>
#include <iostream>

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


    // B. The Portable Communication Logic
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



    // Cleanup
    cudaFree(d_buffer);
    cudaStreamDestroy(stream);
    MPI_Finalize();
    return 0;
}
