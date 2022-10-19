#include <mpi.h>

#include "self_profile.h"
int vftr_MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status) {
   SELF_PROFILE_START_FUNCTION;
   // there is no timing needed but we keep the wrapper here
   // for the instrumentation
   int retVal = PMPI_Iprobe(source, tag, comm, flag, status);
   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
