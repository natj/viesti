







int main() {

  // backend: 1
  // initialize MPI
  // create communicator
  auto comm = GpuComm::Communicator::create(GpuComm::Backend::MPI);

  // backend: 2
  // alternative backend:
  // - same as before but launch NCCL as well
  // - launch a stream inside
  //auto comm = GpuComm::Communicator::create(GpuComm::Backend::NV);

  // MPI: return rank
  // NV: return rank
  int rank = comm->rank();

  //--------------------------------------------------
  // initialize data on a GPU
  size_t count = 1024;
  size_t size_bytes = count * sizeof(float);
  float *buf_A, *buf_B, *buf_C;
  cudaMalloc(&buf_A, size_bytes); // test buffer for put call 1
  cudaMalloc(&buf_B, size_bytes); // test buffer for put call 2
  cudaMalloc(&buf_C, size_bytes); // test buffer for send/recv

  if (rank == 0) {
    std::vector<float> h_A(count, 1.0f);
    std::vector<float> h_B(count, 2.0f);
    std::vector<float> h_C(count, 3.0f);
    cudaMemcpy(buf_A, h_A.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf_B, h_B.data(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(buf_C, h_C.data(), size_bytes, cudaMemcpyHostToDevice);
  } else {
    cudaMemset(buf_A, 0, size_bytes);
    cudaMemset(buf_B, 0, size_bytes);
    cudaMemset(buf_C, 0, size_bytes);
  }
  //--------------------------------------------------
  // library has two message modes
  // - send/recv
  // - put (this needs the buffer to be registered)

  // register buffers A and B for one-sided put
  // last parameter is an id for the buffer
  // MPI: each registration adds a buffer to MPI_Win
  // NV: does nothing (one-sided is not implemented in NV backend)
  comm->register_buffer(buf_A, size_bytes, 10);
  comm->register_buffer(buf_B, size_bytes, 20);

  //--------------------------------------------------
  // mode 1: Put test

  // start epoch where one-sided comm is allowed
  // MPI: MPI_Win_lock 
  // NV: not implemented
  comm->lock_buffer();

  if (rank == 0) {
    // put data into previously defined buffer
    // MPI: MPI_Rput
    // NV: not implemented
    auto req = comm->put(buf_C, count, GpuComm::DataType::FLOAT, 1, tag);

    // wait until local data has been copied away 
    // MPI: MPI_Wait
    // NV: not implemented
    req.wait();

    // calculations on buf_C can safely continue here
  }

  // end epoch where one-sided comm is allowed
  // MPI: MPI_Win_unlock
  // NV: not implemented
  comm->unlock_buffer();

  // DONE: buf_C is now synchronized across ranks



  //--------------------------------------------------
  // mode 2: send/recv test 

  // send/recv communication starts
  // MPI: do nothing
  // NV: ncclGroupStart
  comm->start_sendrecv();

  std::vector<std::unique_ptr<GpuComm::Request>> requests;

  int tag = 666; // dummy tag to enable separation 
  if (rank == 0) {

    // MPI: isend
    // NV: ncclSend
    requests.push_back(comm->send(buf_C, count, GpuComm::DataType::FLOAT, 1, tag));
  } 
  else if (rank == 1) {
    // MPI: irecv
    // NV: ncclRecv
    requests.push_back(comm->recv(buf_C, count, GpuComm::DataType::FLOAT, 0, tag));
  }

  // send/recv communication ends
  // MPI: do nothing
  // NV: ncclGroupEnd, submits job to GPU
  comm->end_sendrecv(); 

  // wait for completion:
  // MPI: MPI_Wait
  // NV: cudaEventSynchronize
  for(auto& req : requests) {
        req->wait();
  }

  if (rank == 1) std::cout << "Rank 1 finished send/recv." << std::endl;


  //--------------------------------------------------
  // clean up
  cudaFree(buf_A);
  cudaFree(buf_B);
  cudaFree(buf_C);
  cudaStreamDestroy(stream);
    
  return 0;
}
