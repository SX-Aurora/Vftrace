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

  unsigned long long addrs [6];
  function_t *func1 = vftr_new_function (NULL, "init", NULL, false);
  function_t *func2 = vftr_new_function ((void*)addrs, "func2", func1, false);
  function_t *func3 = vftr_new_function ((void*)(addrs + 1), "func3", func1, false);	
  function_t *func4 = vftr_new_function ((void*)(addrs + 2), "func4", func3, false);
  function_t *func5 = vftr_new_function ((void*)(addrs + 3), "func5", func2, false);
  function_t *func6 = vftr_new_function ((void*)(addrs + 4), "func6", func2, false);
  function_t *func7 = vftr_new_function ((void*)(addrs + 5), "func4", func6, false);
  vftr_write_stack_ascii (stdout, 0.0, func1, "", 0);
  vftr_write_stack_ascii (stdout, 0.0, func2, "", 0);
  vftr_write_stack_ascii (stdout, 0.0, func3, "", 0);
  vftr_write_stack_ascii (stdout, 0.0, func4, "", 0);
  vftr_write_stack_ascii (stdout, 0.0, func5, "", 0);
  vftr_write_stack_ascii (stdout, 0.0, func6, "", 0);
  vftr_write_stack_ascii (stdout, 0.0, func7, "", 0);

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
