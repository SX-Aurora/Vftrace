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

  fprintf (stdout, "Check MPI rank and size received from environment variables\n");		
  int mpi_rank, mpi_size;
  vftr_get_mpi_info (&mpi_rank, &mpi_size);
  fprintf (stdout, "Rank: %d\n", mpi_rank);
  fprintf (stdout, "Size: %d\n", mpi_size);

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
