#include "vftr_setup.h"

int main (int argc, char **argv) {

#ifdef _MPI
  PMPI_Init(&argc, &argv);
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
