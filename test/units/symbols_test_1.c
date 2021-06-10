#include "vftr_symbols.h"
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

  if (argc != 3) {
    printf ("symbols_test_1: Invalid input - %d arguments\n", argc);
    printf ("require: ./symbols_test <input_file> <output_file>\n");
    return 1;
  }

  char *filename_in = argv[1];
  FILE *fp_in = fopen (filename_in, "r");
  if (fp_in == NULL) {
     printf ("File %s not found!\n", filename_in);
     return 1;
  }
  vftr_nsymbols = 0;
  vftr_get_library_symtab ("", fp_in, 0L, 0);
  vftr_symtab = (symtab_t **) malloc (vftr_nsymbols * sizeof(symtab_t*));

  vftr_nsymbols = 0;
  rewind(fp_in);
  vftr_get_library_symtab("", fp_in, 0L, 1);

  char *filename_out = argv[2];
  FILE *fp_out = fopen (filename_out, "w");
  vftr_print_symbol_table(fp_out, false);
  
  free (vftr_symtab);
  fclose (fp_out);
  fclose (fp_in);
#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
