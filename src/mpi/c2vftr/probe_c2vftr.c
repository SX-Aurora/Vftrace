#ifdef _MPI
#include <mpi.h>

#include "probe.h"

int vftr_MPI_Probe_c2vftr(int source, int tag, MPI_Comm comm, MPI_Status *status) {
   return vftr_MPI_Probe(source, tag, comm, status);
}

#endif
