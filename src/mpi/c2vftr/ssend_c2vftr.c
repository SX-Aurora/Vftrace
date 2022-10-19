#ifdef _MPI
#include <mpi.h>

#include "ssend.h"

int vftr_MPI_Ssend_c2vftr(const void *buf, int count, MPI_Datatype datatype,
                          int dest, int tag, MPI_Comm comm) {
   return vftr_MPI_Ssend(buf, count, datatype, dest, tag, comm);
}

#endif
