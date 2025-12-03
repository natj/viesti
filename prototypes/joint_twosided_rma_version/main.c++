#include "gpu_comm.hpp"
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#define CUDA_CHECK(cmd) { cudaError_t err = cmd; if (err != cudaSuccess) { printf("CUDA Error: %s\n", cudaGetErrorString(err)); exit(1); } }

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Setup a stream for NCCL operations
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Select Backend
  GpuComm::Backend backend = GpuComm::Backend::MPI; 
  // GpuComm::Backend backend = GpuComm::Backend::NV;

  // Initialize data on GPU
  size_t count = 1024;
  size_t size_bytes = count * sizeof(float);
  float *buf_A, *buf_B, *buf_C;
  CUDA_CHECK(cudaMalloc(&buf_A, size_bytes)); 
  CUDA_CHECK(cudaMalloc(&buf_B, size_bytes)); 
  CUDA_CHECK(cudaMalloc(&buf_C, size_bytes)); 

  // Init data
  std::vector<float> h_A(count, 1.0f);
  std::vector<float> h_B(count, 2.0f);
  std::vector<float> h_C(count, rank == 0 ? 100.0f : 0.0f);
  
  CUDA_CHECK(cudaMemcpy(buf_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf_B, h_B.data(), size_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf_C, h_C.data(), size_bytes, cudaMemcpyHostToDevice));

  // Create Communicator
  auto comm = GpuComm::Communicator::create(backend, MPI_COMM_WORLD, stream);

  // --------------------------------------------------
  // ONE SIDED TEST (Only works if Backend == MPI)
  // --------------------------------------------------
  int buf_id_A = 10;
  comm->register_buffer(buf_A, size_bytes, buf_id_A);
  
  MPI_Barrier(MPI_COMM_WORLD); // Ensure all registered before locking

  if (backend == GpuComm::Backend::MPI) {
      int target = 1; 
      
      // We lock the Specific Buffer on the Target Rank
      comm->lock_buffer(target, buf_id_A);

      if (rank == 0) {
        int tag = 0;
        // Put local buf_C into remote Rank 1's buf_A (ID 10)
        auto req = comm->put(buf_C, count, GpuComm::DataType::FLOAT, target, buf_id_A, tag);
        req->wait();
        std::cout << "Rank 0 Put data into Rank 1 buffer." << std::endl;
      }

      comm->unlock_buffer(target, buf_id_A);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // --------------------------------------------------
  // SEND / RECV TEST
  // --------------------------------------------------
  
  comm->start_sendrecv();

  std::vector<std::unique_ptr<GpuComm::Request>> requests;
  int tag = 666;

  if (rank == 0) {
    requests.push_back(comm->send(buf_C, count, GpuComm::DataType::FLOAT, 1, tag));
  } else if (rank == 1) {
    requests.push_back(comm->recv(buf_C, count, GpuComm::DataType::FLOAT, 0, tag));
  }

  comm->end_sendrecv();

  for(auto& req : requests) {
      req->wait();
  }
  
  if (rank == 1) std::cout << "Rank 1 finished recv." << std::endl;

  // Cleanup
  comm.reset(); // Destroy communicator before MPI_Finalize
  CUDA_CHECK(cudaFree(buf_A));
  CUDA_CHECK(cudaFree(buf_B));
  CUDA_CHECK(cudaFree(buf_C));
  CUDA_CHECK(cudaStreamDestroy(stream));

  MPI_Finalize();
  return 0;
}
