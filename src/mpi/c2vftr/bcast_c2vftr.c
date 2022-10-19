#ifdef _MPI
#include <mpi.h>

#include "bcast.h"

int vftr_MPI_Bcast_c2vftr(void *buffer, int count, MPI_Datatype datatype,
                          int root, MPI_Comm comm) {
   // determine if inter or intra communicator
   int isintercom;
   PMPI_Comm_test_inter(comm, &isintercom);
   if (isintercom) {
      return vftr_MPI_Bcast_intercom(buffer, count, datatype, root, comm);
   } else {
      return vftr_MPI_Bcast(buffer, count, datatype, root, comm);
   }
}

#endif
