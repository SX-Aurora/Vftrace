#include <mpi.h>

#include "sync_time.h"

void vftr_estimate_sync_time_F2C(char *routine_name, MPI_Fint *comm_f) {
   MPI_Comm comm_c;
   comm_c = PMPI_Comm_f2c(*comm_f);
   vftr_estimate_sync_time(routine_name, comm_c);
}
