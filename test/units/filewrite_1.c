#include "vftr_environment.h"
#include "vftr_filewrite.h"
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

  vftr_read_environment();

  fprintf (stdout, "Check the creation of log and vfd file name\n");
  int mpi_rank, mpi_size;
  mpi_rank = 0;
  mpi_size = 1;
  fprintf (stdout, "logfile_name(%d, %d): %s\n", mpi_rank, mpi_size,
  	 vftr_create_logfile_name(mpi_rank, mpi_size, "log"));
  mpi_rank = 11;
  mpi_size = 111;
  fprintf (stdout, "logfile_name(%d, %d): %s\n", mpi_rank, mpi_size,
  	 vftr_create_logfile_name(mpi_rank, mpi_size, "log"));
  fprintf (stdout, "logfile_name(%d, %d): %s\n", mpi_rank, mpi_size,
  	 vftr_create_logfile_name(mpi_rank, mpi_size, "vfd"));

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
