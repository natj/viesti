

NCCL or MPI two sided
send/recv


nvSHMEM vs MPI RMA
put



//--------------------------------------------------
// NCCL
ncclGroupStart()
        ncclSend(buf, count, ncclType(type), peer, comm_, stream);

        ncclRecv(buf, count, ncclType(type), peer, comm_, stream);
ncclGroupEnd() // sends job to GPU

cudaEventSynchronize(e);



//--------------------------------------------------
// MPI non-blocking
reqs.append(Isend())
reqs.append(Irecv())
reqs.wait()


//--------------------------------------------------
// MPI RMA / one-sided communication
MPI_Win_attach(win_, ptr, size);
MPI_Allgather(&local_addr, 1, MPI_AINT)

MPI_Win_fence(0, win_);
    MPI_Put(src_buf, count, getMpiType(type), peer, target_disp, count, getMpiType(type), win_);
    // nothing here
MPI_Win_fence(0, win_); 



//--------------------------------------------------
// unified backend
  
// 0. pre-step before simulation
//comm->register_buffer(buf_A, size_bytes, 10);


// A. send communication in timeloop
comm->comm_start();
send(buf_A, count, GpuComm::DataType::FLOAT, 1, 10, stream));


// B. recv communication in timeloop
recv(buf_A, count, GpuComm::DataType::FLOAT, 0, 10, stream));
comm->comm_end(); // Submits work
req->wait();



