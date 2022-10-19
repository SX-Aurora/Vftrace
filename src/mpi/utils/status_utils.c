#include <stdbool.h>

#include <mpi.h>

// mark a MPI_Status as empty
void vftr_empty_mpi_status(MPI_Status *status) {
   // According to the MPI_Standard 3.0 (capter 3.7.3, p.52)
   // A status is empty if its members have the following values:
   // MPI_TAG == MPI_ANY_TAG, MPI_SOURCE == MPI_ANY_SOURCE, MPI_ERROR == MPI_SUCCESS
   status->MPI_TAG = MPI_ANY_TAG;
   status->MPI_SOURCE = MPI_ANY_SOURCE;
   status->MPI_ERROR = MPI_SUCCESS;
   return;
}

// check if a status is empty
bool vftr_mpi_status_is_empty(MPI_Status *status) {
   // According to the MPI_Standard 3.0 (capter 3.7.3, p.52)
   // a status is empty if its members have the following values:
   // MPI_TAG == MPI_ANY_TAG, MPI_SOURCE == MPI_ANY_SOURCE, MPI_ERROR == MPI_SUCCESS
   return (status->MPI_TAG == MPI_ANY_TAG &&
           status->MPI_SOURCE == MPI_ANY_SOURCE &&
           status->MPI_ERROR == MPI_SUCCESS);
}
