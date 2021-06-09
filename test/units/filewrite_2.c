#include "vftr_setup.h"
#include "vftr_functions.h"
#include "vftr_filewrite.h"
#include "vftr_stacks.h"
#include "vftr_hooks.h"
#include "vftr_environment.h"

int main (int argc, char **argv) {

#ifdef _MPI
  PMPI_Init(&argc, &argv);
#endif

  vftr_read_environment();

  unsigned long long addrs [6];
  unsigned long long vftr_test_runtime = 0;
  function_t *func1 = vftr_new_function (NULL, "init", NULL, false);
  function_t *func2 = vftr_new_function ((void*)addrs, "func2", func1, false);
  function_t *func3 = vftr_new_function ((void*)(addrs + 1), "func3", func1, false);	
  function_t *func4 = vftr_new_function ((void*)(addrs + 2), "func4", func3, false);
  function_t *func5 = vftr_new_function ((void*)(addrs + 3), "func5", func2, false);
  function_t *func6 = vftr_new_function ((void*)(addrs + 4), "func6", func2, false);
  function_t *func7 = vftr_new_function ((void*)(addrs + 5), "func4", func6, false);
  vftr_normalize_stacks();
  for (int i = 0; i < vftr_stackscount; i++) {
  	vftr_func_table[i]->prof_current.calls = i + 1;
  	vftr_func_table[i]->prof_current.cycles = 0;
  	vftr_func_table[i]->prof_previous.cycles = 0;
  	vftr_func_table[i]->prof_current.time_excl = (long long)(i+1) * 100000;
  	vftr_func_table[i]->prof_previous.time_excl = (long long)(i+1) * 90000;
  	vftr_func_table[i]->prof_current.time_incl =
  		2 * vftr_func_table[i]->prof_current.time_excl;
  	vftr_func_table[i]->prof_previous.time_incl =
  		2 * vftr_func_table[i]->prof_previous.time_excl;
  	vftr_test_runtime += vftr_func_table[i]->prof_current.time_excl
  			   - vftr_func_table[i]->prof_previous.time_excl;
  }
  
  vftr_profile_wanted = true;
  vftr_mpisize = 1;
  vftr_overhead_usec = 0;
#ifdef _MPI
  vftr_mpi_overhead_usec = 0;
#endif
  int n_func_indices;
  vftr_print_profile (stdout, NULL, &n_func_indices, vftr_test_runtime, 0, NULL);

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
