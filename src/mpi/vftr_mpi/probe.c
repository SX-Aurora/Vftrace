#include <mpi.h>

#include "self_profile.h"
int vftr_MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status) {
   SELF_PROFILE_START_FUNCTION;
   // there is no timing needed but we keep the wrapper here
   // for the instrumentation
   int retVal = PMPI_Probe(source, tag, comm, status);
   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
