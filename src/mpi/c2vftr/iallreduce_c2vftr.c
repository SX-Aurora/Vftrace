#ifdef _MPI
#include <mpi.h>

#include "mpi_buf_addr_const.h"
#include "iallreduce.h"

int vftr_MPI_Iallreduce_c2vftr(const void *sendbuf, void *recvbuf, int count,
                               MPI_Datatype datatype, MPI_Op op, MPI_Comm comm,
                               MPI_Request *request) {
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Iallreduce_intercom(sendbuf, recvbuf,
                                          count, datatype, op,
                                         comm, request);
   } else {
      if (vftr_is_C_MPI_IN_PLACE(sendbuf)) {
         return vftr_MPI_Iallreduce_inplace(sendbuf, recvbuf,
                                            count, datatype, op,
                                            comm, request);
      } else {
         return vftr_MPI_Iallreduce(sendbuf, recvbuf,
                                    count, datatype, op,
                                    comm, request);
      }
   }
}

#endif
