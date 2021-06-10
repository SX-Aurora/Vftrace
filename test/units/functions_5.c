#include "vftr_functions.h"
#include "vftr_stacks.h"
#include "vftr_environment.h"

int main (int argc, char **argv) {

#ifdef _MPI
  PMPI_Init(&argc, &argv);
#endif

  vftr_read_environment();

  unsigned long long addrs [6];
  function_t *func1 = vftr_new_function (NULL, "INIT", NULL, false);
  function_t *func2 = vftr_new_function ((void*)addrs, "fUnC2", func1, false);
  function_t *func3 = vftr_new_function ((void*)(addrs + 1), "FUnc3", func1, false);	
  function_t *func4 = vftr_new_function ((void*)(addrs + 2), "func4", func3, false);
  function_t *func5 = vftr_new_function ((void*)(addrs + 3), "fUNC2", func4, false);
  vftr_write_function_indices (stdout, "init", true);
  vftr_write_function_indices (stdout, "func2", true);
  vftr_write_function_indices (stdout, "func3", true);
  vftr_write_function_indices (stdout, "func4", true);

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
