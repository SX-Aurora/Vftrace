#ifdef _MPI
#include <mpi.h>

#include "send.h"

int vftr_MPI_Send_c2vftr(const void *buf, int count, MPI_Datatype datatype,
                         int dest, int tag, MPI_Comm comm) {
   return vftr_MPI_Send(buf, count, datatype, dest, tag, comm);
}

#endif
