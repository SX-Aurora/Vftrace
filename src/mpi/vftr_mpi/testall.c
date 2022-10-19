#include <mpi.h>

#include <stdbool.h>

#include "self_profile.h"
#include "requests.h"

int vftr_MPI_Testall(int count, MPI_Request array_of_requests[],
                     int *flag, MPI_Status array_of_statuses[]) {
   SELF_PROFILE_START_FUNCTION;
   int retVal;
   int tmpflag;

   // set the return flag to true
   *flag = true;
   // It will be returned true if all communications are completed
   for (int ireq=0; ireq<count; ireq++) {
      if (array_of_statuses == MPI_STATUSES_IGNORE) {
         retVal = PMPI_Request_get_status(array_of_requests[ireq],
                                          &tmpflag,
                                          MPI_STATUS_IGNORE);
      } else {
         retVal = PMPI_Request_get_status(array_of_requests[ireq],
                                          &tmpflag,
                                          array_of_statuses+ireq);
      }
      // accumulate truthness of the individual requests
      // as soon as one if false everything is
      *flag = (*flag) && tmpflag;
   }
   // clear completed communications from the list of open requests
   vftr_clear_completed_requests_from_test();

   if (flag) {
   // If all communications are completed
   // run Testall to modify the requests appropriately
      retVal = PMPI_Testall(count, array_of_requests, flag, array_of_statuses);
   }

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
