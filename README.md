# Viesti

**Viesti** is a lightweight, high-performance C++ communication abstraction library designed for GPU-centric HPC applications. It provides a unified, intuitive interface for GPU-to-GPU message passing, abstracting away the complexities of mixing underlying backends like MPI and RCCL.

The library supports simultaneous usage of different backends for One-Sided (RMA) and Two-Sided (P2P) communication, selectable entirely at compile-time to avoid runtime overhead.

## Key Features

* **Unified Interface**: A single `Communicator` class handles both MPI and RCCL backends seamlessly.
* **Zero-Cost Abstraction**: Backends are selected via preprocessor flags, ensuring no virtual function overhead in critical paths.
* **Hybrid Modes**: Mix and match backends (e.g., use MPI for RMA windows and RCCL for high-bandwidth P2P transfers).
* **Modern C++**: Type-safe, templated API with automatic data type deduction (`float`, `double`, `int`).
* **Unified Request Management**: A single concrete `Request` class handles both MPI requests and HIP events/streams.


## Compilation

The library relies on **mutually exclusive flags** to configure the RMA and P2P backends. You must define at least one RMA backend and one P2P backend.

### Backend Flags

| Flag | Category | Description |
| :--- | :--- | :--- |
| `-DMPI_RMA` | RMA | Enables standard MPI One-Sided communication (`MPI_Put`, `MPI_Win_lock`). |
| `-DMPI_P2P` | P2P | Enables standard MPI Point-to-Point (`MPI_Isend`, `MPI_Irecv`). |
| `-DHIP_P2P` | P2P | Enables RCCL backend (`ncclSend`, `ncclRecv`) on HIP streams. |


