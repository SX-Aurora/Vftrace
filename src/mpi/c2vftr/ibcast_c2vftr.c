#ifdef _MPI
#include <mpi.h>

#include "ibcast.h"

int vftr_MPI_Ibcast_c2vftr(void *buffer, int count, MPI_Datatype datatype,
                           int root, MPI_Comm comm, MPI_Request *request) {
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Ibcast_intercom(buffer, count, datatype, root, comm, request);
   } else {
      return vftr_MPI_Ibcast(buffer, count, datatype, root, comm, request);
   }
}

#endif
