#include <mpi.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

#include "viesti.h"


// small test application to test the library
int main(int argc, char** argv) {
  
  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  // --- Display Active Configuration ---
  std::cout << "viesti Configuration:" << std::endl;
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
  viesti::Communicator comm(&argc, &argv, stream);
  int rank = comm.rank();

  // --- Data Init ---
  size_t count = 1024;
  size_t size_bytes = count * sizeof(float);
  float *buf_A_dev, *buf_B_dev, *buf_C_dev;
  
  CHECK_HIP(hipMalloc(&buf_A_dev, size_bytes)); 
  CHECK_HIP(hipMalloc(&buf_B_dev, size_bytes)); 
  CHECK_HIP(hipMalloc(&buf_C_dev, size_bytes)); 

  std::vector<float> buf_A_host(count, rank == 0 ? 1.0f : 0.0f);
  std::vector<float> buf_B_host(count, rank == 0 ? 2.0f : 0.0f);
  std::vector<float> buf_C_host(count, rank == 0 ? 3.0f : 0.0f);
  
  CHECK_HIP(hipMemcpy(buf_A_dev, buf_A_host.data(), size_bytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(buf_B_dev, buf_B_host.data(), size_bytes, hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(buf_C_dev, buf_C_host.data(), size_bytes, hipMemcpyHostToDevice));


   
  // --------------------------------------------------
  // TEST 1: PASSIVE RMA COMM
  //         receiver does nothing; syncs with barrier (or something else)
  // --------------------------------------------------
  int buf_id_A = 10;
  int buf_id_B = 20;

  // register buffers
  comm.register_buffer(buf_A_dev, size_bytes, buf_id_A);

  comm.barrier(); // global sync

  int origin_rank = 0; 
  int target_rank = 1; 
  if (rank == origin_rank) { // only origin rank does comm

    // lock the target windows 
    comm.lock_buffer(target_rank, buf_id_A);

    std::vector<viesti::Request> reqs_rma;
    reqs_rma.push_back( comm.put(buf_A_dev, count, target_rank, buf_id_A) );

    for(auto& req : reqs_rma) req.wait();
    std::cout << rank << ": rank 0 Put data (RMA) into A and B buffers." << std::endl;

    // rank 0 can now modify data
  
    // flush actual data into target registers
    comm.unlock_buffer(target_rank, buf_id_A);

    // rank 1 data is now modified
  }

  comm.barrier(); // global sync

  // dev2host
  CHECK_HIP(hipMemcpy(buf_A_host.data(), buf_A_dev, size_bytes, hipMemcpyDeviceToHost));
  std::cout << rank << ": finished passive put into A buffer: " << buf_A_host[0] << std::endl;



  // --------------------------------------------------
  // TEST 2: ACTIVE RMA COMM
  //         sender and receiver do an explicit handshake 
  //
  //
  // WORKING NOTE: this reduces to send & recv and can be implemented as a new p2p backend.
  //               This is sketched in prototypes/idea9.c++.
  //
  // --------------------------------------------------
  //origin_rank = 0;
  //target_rank = 1;

  //comm.register_buffer(buf_B_dev, size_bytes, buf_id_B); // register buffer B 
  //                                                         
  //// Handshake to ensure Post happens before Start 
  //// (Conceptually safer, though MPI allows loose ordering)
  //comm.barrier(); 

  //// sender (origin)
  //if (rank == origin_rank) { 
  //  // "I am starting access to target using Window B"
  //  comm.start_rma_send(target_rank, buf_id_B); // MPI_Win_start

  //  // queue the Put
  //  auto req = comm.put(buf_B_dev, count, target_rank, buf_id_B);
  //  //req.wait() // not needed

  //  // "i am done. Ensure data is sent."
  //  // This blocks until local buffer is safe to reuse.
  //  comm.end_rma_send(buf_id_B);  // MPI_Win_complete
  //}
  //
  //// receiver (destination)
  //if (rank == target_rank) { 
  //  // "I allow origin to access my Window B"
  //  comm.start_rma_recv(origin_rank, buf_id_B); // MPI_win_post

  //  // "I wait until origin is finished"
  //  // This blocks until data has fully arrived.
  //  comm.end_rma_recv(buf_id_B);  // MPI_win_wait
  //}

  //// At this point, Rank 1 (target) definitely has the data in buf_B_dev


  //// communication routine ends-----------
  //comm.barrier(); // global sync

  //// dev2host
  //CHECK_HIP(hipMemcpy(buf_B_host.data(), buf_B_dev, size_bytes, hipMemcpyDeviceToHost));
  //std::cout << rank << ": finished passive put into B buffer: " << buf_B_host[0] << std::endl;



  // --------------------------------------------------
  // TEST 3: SEND / RECV
  // --------------------------------------------------
  comm.start_sendrecv();
  // we can now send and receive inside of this epoch
  //NCCL treats all calls between start and end as a single call to many devices.

  //std::cout << rank << ": start " << buf_C_host[0] << std::endl;
  std::vector<viesti::Request> reqs_p2p;
  int tag = 42;

  if (rank == 0) {
    reqs_p2p.push_back(comm.send(buf_C_dev, count, 1, tag));
    //std::cout << rank << ": send " << buf_C_host[0] << std::endl;
  } else if (rank == 1) {
    reqs_p2p.push_back(comm.recv(buf_C_dev, count, 0, tag));
    //std::cout << rank << ": recv " << buf_C_host[0] << std::endl;
  }

  comm.end_sendrecv(); // close comm epoch and launch tasks on GPU
  //std::cout << rank << ": end " << buf_C_host[0] << std::endl;

  for(auto& req : reqs_p2p) req.wait(); // wait until messages have been transferred
  //std::cout << rank << ": end wait" << buf_C_host[0] << std::endl;
  

  CHECK_HIP(hipMemcpy(buf_C_host.data(), buf_C_dev, size_bytes, hipMemcpyDeviceToHost));

  //std::cout << rank << ": memcopy" << buf_C_host[0] << std::endl;
  //std::cout << rank << ": finished send/recv into C buffer: " << buf_C_host[0] << std::endl;

  //--------------------------------------------------
  CHECK_HIP(hipFree(buf_A_dev));
  CHECK_HIP(hipFree(buf_B_dev));
  CHECK_HIP(hipFree(buf_C_dev));
  CHECK_HIP(hipStreamDestroy(stream));

  return 0;
}
