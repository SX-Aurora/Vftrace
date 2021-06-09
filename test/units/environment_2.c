#include "vftr_environment.h"

int main (int argc, char **argv) {

#ifdef _MPI
  PMPI_Init(&argc, &argv);
#endif

  // Check if the environment advisor works
  putenv ("VFTR_OF=yes"); // Should be VFTR_OFF
  putenv ("VFTR_TRUNCATE=yes"); // Should be VFTR_PROF_TRUNCATE

  vftr_read_environment ();
  vftr_assert_environment ();
  vftr_free_environment ();

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
