#include "vftr_functions.h"
#include "vftr_stacks.h"
#include "vftr_environment.h"
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

  vftr_read_environment();

  fprintf (stdout, "Initial vftr_stackscount: %d\n", vftr_stackscount);
  
  int i0 = vftr_stackscount;
  if (i0 > 0) fprintf (stdout, "Check additional MPI entries:\n");
  for (int i = 0; i < i0; i++) {
  	vftr_write_function (stdout, vftr_func_table[i], true);
  }
  
  unsigned long long addrs [6];
  function_t *func1 = vftr_new_function (NULL, "init_vftr", NULL, false);
  function_t *func2 = vftr_new_function ((void*)addrs, "func2", func1, false);
  function_t *func3 = vftr_new_function ((void*)(addrs + 1), "func3", func1, false);	
  function_t *func4 = vftr_new_function ((void*)(addrs + 2), "func4", func3, false);
  function_t *func5 = vftr_new_function ((void*)(addrs + 3), "func5", func2, false);
  function_t *func6 = vftr_new_function ((void*)(addrs + 4), "func6", func2, false);
  function_t *func7 = vftr_new_function ((void*)(addrs + 5), "func4", func6, false);
  fprintf (stdout, "Check test entries:\n");
  for (int i = i0; i < vftr_stackscount; i++) {
  	vftr_write_function(stdout, vftr_func_table[i], true);
  }
  fprintf (stdout, "Test if callee pointer is changed properly\n");
  func2->callee = func6;
  vftr_write_function (stdout, func2, true);
  fprintf (stdout, "vftr_func_table_size: %d\n", vftr_func_table_size);
  fprintf (stdout, "vftr_stackscount: %d\n", vftr_stackscount);
  fprintf (stdout, "Check functions registered in function table: \n");
  for (int i = i0; i < vftr_stackscount; i++) {
  	vftr_write_function(stdout, vftr_func_table[i], true);
  }

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
