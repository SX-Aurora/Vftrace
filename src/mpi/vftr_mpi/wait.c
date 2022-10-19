#include <mpi.h>

#include <stdbool.h>

#include "self_profile.h"
#include "requests.h"

int vftr_MPI_Wait(MPI_Request *request, MPI_Status *status) {
   SELF_PROFILE_START_FUNCTION;
   int retVal;

   // loop until the communication corresponding to the request is completed
   int flag = false;
   while (!flag) {
      // check if the communication is finished
      retVal = PMPI_Request_get_status(*request,
                                       &flag,
                                       status);
      // either the communication is completed, or not
      // other communications might be completed in the background
      // clear those from the list of open requests
      vftr_clear_completed_requests_from_wait();
   }
   // Properly set the request and status variable
   retVal = PMPI_Wait(request, status);

   SELF_PROFILE_END_FUNCTION;
   return retVal;
}
