#include "vftr_scenarios.h"
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

  if (argc != 2) {
    printf ("scenarios_1: Missing input file argument.\n");
    return 1;
  }

  FILE *fp_in = fopen (argv[1], "r");
  if (fp_in == NULL) {
    printf ("Scenario file %s not found!\n", argv[1]);
    return 1;
  }

  // The input file contains wrong syntax. The test is therefore XFAIL.
  vftr_read_scenario_file("", fp_in);

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
