#include <mpi.h>

#include <stdlib.h>
#include <stdbool.h>

#include "self_profile.h"
#include "requests.h"

int vftr_MPI_Waitall(int count, MPI_Request array_of_requests[],
                     MPI_Status array_of_statuses[]) {
   SELF_PROFILE_START_FUNCTION;
   if (count <= 0) {
      SELF_PROFILE_END_FUNCTION;
      return MPI_SUCCESS;
   }

   // loop while at least one request is not completed
   int *req_completed = (int*) malloc(count*sizeof(int));
   for (int ireq=0; ireq<count; ireq++) {
      req_completed[ireq] = false;
   }
   int tmpflag;
   bool all_completed = false;
   while (!all_completed) {
      all_completed = true;
      // loop over all requests
      for (int ireq=0; ireq<count; ireq++) {
         if (!req_completed[ireq]) {
            // check if the communication associated with the request
            // is completed
            PMPI_Request_get_status(array_of_requests[ireq],
                                    &tmpflag,
                                    MPI_STATUS_IGNORE);
            // if not completed
            req_completed[ireq] = tmpflag;
            if (!(req_completed[ireq])) {
               all_completed = false;
            }
         }
      }
      vftr_clear_completed_requests_from_wait();
   }

   free(req_completed);
   req_completed = NULL;

   int retVal = PMPI_Waitall(count, array_of_requests, array_of_statuses);
   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
