#include <stdlib.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "configuration.h"
#include "logfile_common.h"
#include "vfdfiles.h"

#ifdef _MPI
#include <mpi.h>
#endif

int main(int argc, char **argv) {
   INIT_SELF_PROF_VFTRACE;
#if defined(_MPI)
   PMPI_Init(&argc, &argv);
#else
   (void) argc;
   (void) argv;
#endif

   config_t config;
   config = vftr_read_config();

  fprintf (stdout, "Check the creation of log and vfd file name\n");
  char *logfile_name = NULL;
  int mpi_rank, mpi_size;
  mpi_rank = 0;
  mpi_size = 1;

  logfile_name = vftr_get_logfile_name(config, NULL, mpi_rank, mpi_size);
  fprintf(stdout, "logfile_name(%d, %d): %s\n", mpi_rank, mpi_size, logfile_name);
  free(logfile_name);

  mpi_rank = 11;
  mpi_size = 111;

  logfile_name = vftr_get_logfile_name(config, NULL, mpi_rank, mpi_size);
  fprintf(stdout, "logfile_name(%d, %d): %s\n", mpi_rank, mpi_size, logfile_name);
  free(logfile_name);

  logfile_name = vftr_get_vfdfile_name(config, mpi_rank, mpi_size);
  fprintf(stdout, "logfile_name(%d, %d): %s\n", mpi_rank, mpi_size, logfile_name);
  free(logfile_name);

   vftr_config_free(&config);
#ifdef _MPI
  PMPI_Finalize();
#endif

  FINALIZE_SELF_PROF_VFTRACE;
  return 0;
}
