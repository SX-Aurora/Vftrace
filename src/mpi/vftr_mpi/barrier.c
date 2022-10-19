#include <mpi.h>

#include "self_profile.h"

int vftr_MPI_Barrier(MPI_Comm comm) {
   SELF_PROFILE_START_FUNCTION;
   // there is no timing needed but we keep the wrapper here
   // for the instrumentation
   int retVal = PMPI_Barrier(comm);
   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
