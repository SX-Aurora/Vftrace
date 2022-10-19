#ifdef _MPI
#include <mpi.h>

#include "sendrecv_replace.h"

int vftr_MPI_Sendrecv_replace_c2vftr(void *buf, int count, MPI_Datatype datatype,
                                     int dest, int sendtag, int source, int recvtag,
                                     MPI_Comm comm, MPI_Status *status) {
   return vftr_MPI_Sendrecv_replace(buf, count, datatype, dest, sendtag,
                                    source, recvtag, comm, status);
}

#endif
