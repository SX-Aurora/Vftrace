#include <stdlib.h>
#include <stdio.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "configuration.h"
#include "configuration_assert.h"

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
   vftr_config_assert(stderr, config);
   vftr_config_free(&config);

#ifdef _MPI
   PMPI_Finalize();
#endif

   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
