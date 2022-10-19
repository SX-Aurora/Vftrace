#ifdef _MPI
#include <mpi.h>

#include "bsend.h"

int vftr_MPI_Bsend_c2vftr(const void *buf, int count, MPI_Datatype datatype,
                          int dest, int tag, MPI_Comm comm) {
   return vftr_MPI_Bsend(buf, count, datatype, dest, tag, comm);
}

#endif
