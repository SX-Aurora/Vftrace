#include <mpi.h>

#include "self_profile.h"
#include "requests.h"

int vftr_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) {
   SELF_PROFILE_START_FUNCTION;
   // do not call MPI_Test immediately!
   // If the communication is successfull MPI_Test destroys the Reqeust
   // Hence, no chance of properly clearing the communication
   // from the open request list
   // MPI_Request_get_status is a non destructive check.
   int retVal = PMPI_Request_get_status(*request, flag, status);

   if (*flag) {
      // Communication is done.
      // Clear finished communications from the open request list
      vftr_clear_completed_requests_from_test();
      // Now that the danger of deleating needed requests is banned
      // actually call MPI_Test
      retVal = PMPI_Test(request, flag, status);
   }

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
