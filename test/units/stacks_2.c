// This test creates four artificial local stack trees on each rank, which are merged
// using vftr_normalize_stacks. We print out each local stack and the corresponding global stack list.
// This test is only executed when MPI is active. Thus, there is no distinction for serial and parallel
// versions here (no #ifdef _MPI).

#include "vftr_hooks.h"
#include "vftr_functions.h"
#include "vftr_stacks.h"
#include "vftr_environment.h"
#include "vftr_setup.h"
#include <mpi.h>

int main (int argc, char **argv) {

  PMPI_Init(&argc, &argv);
  vftr_get_mpi_info (&vftr_mpirank, &vftr_mpisize);

  vftr_read_environment();

  unsigned long long addrs[6];
  function_t *func0 = vftr_new_function (NULL, "init", NULL, false);	
  if (vftr_mpirank == 0) {
  	function_t *func1 = vftr_new_function ((void*)addrs, "func1", func0, false);
  	function_t *func2 = vftr_new_function ((void*)(addrs + 1), "func2", func1, false);
  	function_t *func3 = vftr_new_function ((void*)(addrs + 2), "func3", func2, false);
  	function_t *func4 = vftr_new_function ((void*)(addrs + 3), "func4", func3, false);
  } else if (vftr_mpirank == 1) {
  	function_t *func1 = vftr_new_function ((void*)addrs, "func1", func0, false);
  	function_t *func2 = vftr_new_function ((void*)(addrs + 2), "func3", func1, false);
  	function_t *func3 = vftr_new_function ((void*)(addrs + 1), "func2", func2, false);
  	function_t *func4 = vftr_new_function ((void*)(addrs + 3), "func4", func3, false);
  } else if (vftr_mpirank == 2) {	
  	function_t *func1 = vftr_new_function ((void*)addrs, "func1", func0, false);
  	function_t *func2 = vftr_new_function ((void*)(addrs + 1), "func2", func1, false);
  	function_t *func3 = vftr_new_function ((void*)(addrs + 2), "func2", func2, false);
  	function_t *func4 = vftr_new_function ((void*)(addrs + 3), "func2", func3, false);
  } else if (vftr_mpirank == 3) {
  	function_t *func1 = vftr_new_function ((void*)addrs, "func1", func0, false);
  	function_t *func2 = vftr_new_function ((void*)(addrs + 3), "func4", func1, false);
  } else {
  	fprintf (stdout, "Error: Invalid MPI rank (%d)!\n", vftr_mpirank);
  	return -1;
  }
  
  vftr_environment.logfile_all_ranks->value = true;
  vftr_normalize_stacks();
  vftr_create_global_stack_strings();
  
  // Needs to be set for printing the local stacklist
  vftr_profile_wanted = true;
  for (int i = 0; i < vftr_mpisize; i++) {
  	if (vftr_mpirank == i) {
  		fprintf (stdout, "Local stacklist for rank %d: \n", i);
  		// There is "init" + the four (rank 0 - 2) or two (rank 3) additional functions.
  		int n_functions = vftr_mpirank == 3 ? 3 : 5;
  		vftr_print_local_stacklist (vftr_func_table, stdout, n_functions);
  	        fprintf (stdout, "Global stacklis for rank %d: \n", i);
                // NOTE: vftr_print_global_stacklist only prints the stack IDs which are present on the given rank.
  	        vftr_print_global_stacklist (stdout);
  	}
        // Barrier sorts output
  	PMPI_Barrier (MPI_COMM_WORLD);
  }


  PMPI_Finalize();

  return 0;
}
