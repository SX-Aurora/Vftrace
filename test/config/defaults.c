#include <stdlib.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "configuration.h"
#include "configuration_print.h"

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
   vftr_print_config(stdout, config, true);
   vftr_config_free(&config);

#ifdef _MPI
   PMPI_Finalize();
#endif
   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
