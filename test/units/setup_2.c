#include "vftr_setup.h"
#ifdef _MPI
#include <mpi.h>
#endif

int main (int argc, char **argv) {

#if defined(_MPI)
  PMPI_Init(&argc, &argv);
  vftr_get_mpi_info (&vftr_mpirank, &vftr_mpisize);
#else 
  vftr_mpirank = 0;
  vftr_mpisize = 1;
#endif

  fprintf (stdout, "Check disclaimers\n");
  vftr_print_disclaimer_full (stdout);
  fprintf (stdout, "****************************************\n");
  vftr_print_disclaimer (stdout, true);

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
