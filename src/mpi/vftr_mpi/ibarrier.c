#include <mpi.h>

#include "self_profile.h"

int vftr_MPI_Ibarrier(MPI_Comm comm, MPI_Request *request) {
   SELF_PROFILE_START_FUNCTION;
   // there is no timing needed but we keep the wrapper here
   // for the instrumentation
   int retVal = PMPI_Ibarrier(comm, request);
   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
