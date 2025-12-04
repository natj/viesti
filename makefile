# CXX      = hipcc
SRC      = viesti.c++
TARGET   = viesti

# --- Configuration Flags ---
# Hybrid Mode:  -DMPI_RMA -DHIP_P2P
# Pure MPI:     -DMPI_RMA -DMPI_P2P
DEFS     = -DMPI_RMA -DMPI_P2P
#DEFS     = -DMPI_RMA -DHIP_P2P
#DEFS     = -DHIP_RMA -DHIP_P2P

# --- Libraries ---
# Ensure MPI and RCCL are in your library path or add -L/path/to/libs
# You might need to add -I/usr/include/mpi depending on your setup
#LIBS = -lmpi -lrccl
#LIBS = -lmpi
LIBS = 

FLAGS  = -O2 -std=c++14

all:
	$(CXX) $(FLAGS) $(DEFS) -x hip $(SRC) -o $(TARGET) $(LIBS)

clean:
	rm -f $(TARGET)
