#include <stdlib.h>

#include "self_profile.h"
#include "environment_types.h"
#include "environment.h"

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
   environment_t environment;
   environment = vftr_read_environment();
   vftr_environment_assert(stderr, environment);
   vftr_print_environment(stdout, environment);
   vftr_environment_free(&environment);

#ifdef _MPI
   PMPI_Finalize();
#endif
   FINALIZE_SELF_PROF_VFTRACE;
   return 0;
}
