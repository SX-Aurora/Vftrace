#include <stdbool.h>

#include <mpi.h>

#include "status_utils.h"

// check if a request is active
bool vftr_mpi_request_is_active(MPI_Request request) {
   // According to the MPI_Standard 3.0 (capter 3.7.3, p.52)
   // a request is active if it is neither a null request
   // nor returns an empty status for Request_get_status
   // (the function returns an empty status if it is inactive)

   if (request == MPI_REQUEST_NULL) {
      return false;
   }

   MPI_Status status;
   int flag;
   PMPI_Request_get_status(request, &flag, &status);

   return !vftr_mpi_status_is_empty(&status);
}
