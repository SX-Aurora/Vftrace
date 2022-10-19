#include <mpi.h>

#include <stdbool.h>

#include "self_profile.h"
#include "request_utils.h"
#include "status_utils.h"
#include "requests.h"

int vftr_MPI_Testany(int count, MPI_Request array_of_requests[],
                     int *index, int *flag, MPI_Status *status) {
   SELF_PROFILE_START_FUNCTION;
   // First check if the request array contains at least one active handle
   bool activereqs = false;
   for (int ireq=0; ireq<count; ireq++) {
      if (vftr_mpi_request_is_active(array_of_requests[ireq])) {
         activereqs = true;
         break;
      }
   }
   // if no active request is found return with the following settings
   if (!activereqs) {
      *flag = true;
      *index = MPI_UNDEFINED;
      if (status != MPI_STATUS_IGNORE) {
         vftr_empty_mpi_status(status);
      }
      SELF_PROFILE_END_FUNCTION;
      return MPI_SUCCESS;
   }

   // initialize the index to the default failure value
   *index = MPI_UNDEFINED;
   // loop over all requests and terminate the loop
   // on the first completed communication
   int retVal = 0;
   for (int ireq=0; ireq<count; ireq++) {
      retVal = PMPI_Request_get_status(array_of_requests[ireq],
                                       flag,
                                       status);
      // terminate if a completed communication is found
      if (flag) {
         // record the index of the completed communication
         *index = ireq;
         // clear completed communication from the list of open requests
         vftr_clear_completed_requests_from_test();
         // Mark the request as inactive, or deallocate it.
         retVal = PMPI_Test(array_of_requests+ireq,
                            flag,
                            status);
         break;
      }
   }

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
