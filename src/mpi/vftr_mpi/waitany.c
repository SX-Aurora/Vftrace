#include <mpi.h>

#include <stdbool.h>

#include "self_profile.h"
#include "request_utils.h"
#include "status_utils.h"
#include "requests.h"

int vftr_MPI_Waitany(int count, MPI_Request array_of_requests[],
                     int *index, MPI_Status *status) {
   SELF_PROFILE_START_FUNCTION;
   if (count <= 0) {
      SELF_PROFILE_END_FUNCTION;
      return MPI_SUCCESS;
   }

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
      *index = MPI_UNDEFINED;
      if (status == MPI_STATUS_IGNORE) {
         vftr_empty_mpi_status(status);
      }
      SELF_PROFILE_END_FUNCTION;
      return MPI_SUCCESS;
   }

   // loop until at least one communication of the requests is completed
   int retVal;
   int completed_req = false;
   while (!completed_req) {
      // loop over all requests
      for (int ireq=0; ireq<count; ireq++) {
         int flag;
         // check if the communication associated with the request
         // is completed
         retVal = PMPI_Request_get_status(array_of_requests[ireq],
                                          &flag,
                                          status);
         completed_req = completed_req || flag;
         // either the communication is completed, or not
         // other communications might be completed in the background
         // clear those from the list of open requests
         vftr_clear_completed_requests_from_wait();
         // if this request corresponds to a completed communication
         // leave the loop
         if (flag) {
            // record the index of the finished request
            *index = ireq;
            break;
         }
      }
   }

   // Properly set the request and status variable
   retVal = PMPI_Wait(array_of_requests+(*index), status);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
