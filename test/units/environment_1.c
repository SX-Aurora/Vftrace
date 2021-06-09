#include "vftr_environment.h"

int main (int argc, char **argv) {

#ifdef _MPI
  PMPI_Init(&argc, &argv);
#endif

  if (argc != 2) {
    printf ("environment_1: Invalid input - %d arguments\n", argc);
    printf ("Require: ./environment_1 <output_file>\n");
    return 1;
  }

  char *filename_out = argv[1];
  FILE *fp_out = fopen (filename_out, "w");

  // The basic usage of the environment module:
  putenv ("VFTR_OFF=yes");
  vftr_read_environment();
  vftr_assert_environment();
  vftr_print_environment(fp_out);
  vftr_free_environment();

  // Check each of the different possible data types (except regular expression).
  fprintf (fp_out, "***************************\n");
  putenv ("VFTR_OFF=no");
  putenv ("VFTR_SAMPLING=YES");
  putenv ("VFTR_REGIONS_PRECISE=0");
  putenv ("VFTR_MPI_LOG=on");
  putenv ("VFTR_OUT_DIRECTORY=\"foo/bar\"");
  putenv ("VFTR_BUFSIZE=1234");
  putenv ("VFTR_SAMPLETIME=12.34");
  
  vftr_read_environment ();
  vftr_assert_environment ();
  vftr_print_environment (fp_out);
  vftr_free_environment ();

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
