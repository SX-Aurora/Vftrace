#include "vftr_setup.h"

int main (int argc, char **argv) {

#ifdef _MPI
  PMPI_Init(&argc, &argv);
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
