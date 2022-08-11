#include <stdlib.h>

#include "environment_types.h"
#include "environment.h"
#include "ranklogfile.h"
#include "vfdfiles.h"

#ifdef _MPI
#include <mpi.h>
#endif

int main(int argc, char **argv) {
#if defined(_MPI)
   PMPI_Init(&argc, &argv);
#else
   (void) argc;
   (void) argv;
#endif

   environment_t environment;
   environment = vftr_read_environment();
   vftr_environment_free(&environment);

  fprintf (stdout, "Check the creation of log and vfd file name\n");
  char *logfile_name = NULL;
  int mpi_rank, mpi_size;
  mpi_rank = 0;
  mpi_size = 1;

  logfile_name = vftr_get_ranklogfile_name(environment, mpi_rank, mpi_size);
  fprintf(stdout, "logfile_name(%d, %d): %s\n", mpi_rank, mpi_size, logfile_name);
  free(logfile_name);

  mpi_rank = 11;
  mpi_size = 111;

  logfile_name = vftr_get_ranklogfile_name(environment, mpi_rank, mpi_size);
  fprintf(stdout, "logfile_name(%d, %d): %s\n", mpi_rank, mpi_size, logfile_name);
  free(logfile_name);

  logfile_name = vftr_get_vfdfile_name(environment, mpi_rank, mpi_size);
  fprintf(stdout, "logfile_name(%d, %d): %s\n", mpi_rank, mpi_size, logfile_name);
  free(logfile_name);

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
