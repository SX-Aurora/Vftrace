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

  unsigned long long addrs[6];
  fprintf (stdout, "Initial vftr_stackscount: %d\n", vftr_stackscount);
  function_t *func1 = vftr_new_function (NULL, "init", NULL, false);
  function_t *func2 = vftr_new_function ((void*)addrs, "func2", func1, false);
  function_t *func3 = vftr_new_function ((void*)(addrs + 1), "func3", func1, false);	
  function_t *func4 = vftr_new_function ((void*)(addrs + 2), "func4", func3, false);
  function_t *func5 = vftr_new_function ((void*)(addrs + 3), "func5", func2, false);
  function_t *func6 = vftr_new_function ((void*)(addrs + 4), "func6", func2, false);
  function_t *func7 = vftr_new_function ((void*)(addrs + 5), "func4", func6, false);
  vftr_normalize_stacks();			
  fprintf (stdout, "%s: %d %d\n", func1->name, func1->id, func1->gid);
  fprintf (stdout, "%s: %d %d\n", func2->name, func2->id, func2->gid);
  fprintf (stdout, "%s: %d %d\n", func3->name, func3->id, func3->gid);
  fprintf (stdout, "%s: %d %d\n", func4->name, func4->id, func4->gid);
  fprintf (stdout, "%s: %d %d\n", func5->name, func5->id, func5->gid);
  fprintf (stdout, "%s: %d %d\n", func6->name, func6->id, func6->gid);
  fprintf (stdout, "%s: %d %d\n", func7->name, func7->id, func7->gid);
  fprintf (stdout, "Global stacklist: \n");
  vftr_create_global_stack_strings();
  vftr_print_global_stacklist (stdout);

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}
