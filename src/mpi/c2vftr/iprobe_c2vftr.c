#ifdef _MPI
#include <mpi.h>

#include "iprobe.h"

int vftr_MPI_Iprobe_c2vftr(int source, int tag, MPI_Comm comm,
                           int *flag, MPI_Status *status) {
   return vftr_MPI_Iprobe(source, tag, comm, flag, status);
}

#endif
