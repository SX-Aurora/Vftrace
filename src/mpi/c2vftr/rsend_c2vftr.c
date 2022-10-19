#ifdef _MPI
#include <mpi.h>

#include "rsend.h"

int vftr_MPI_Rsend_c2vftr(const void *buf, int count, MPI_Datatype datatype,
                          int dest, int tag, MPI_Comm comm) {
   return vftr_MPI_Rsend(buf, count, datatype, dest, tag, comm);
}

#endif
