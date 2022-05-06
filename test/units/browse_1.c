#include "vftr_browse.h"
#include "vftr_environment.h"
#include "vftr_stacks.h"
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
  function_t *func1 = vftr_new_function (NULL, "init", NULL, false);
  function_t *func2 = vftr_new_function ((void*)addrs, "MAIN__", func1, false);
  function_t *func3 = vftr_new_function ((void*)(addrs + 1), "A", func1, false);
  function_t *func4 = vftr_new_function ((void*)(addrs + 2), "B", func1, false);
  function_t *func5 = vftr_new_function ((void*)(addrs + 3), "C", func3, false);
  function_t *func6 = vftr_new_function ((void*)(addrs + 4), "C", func4, false);
  vftr_normalize_stacks();
  for (int i = 0; i < vftr_stackscount; i++) {
  	vftr_func_table[i]->prof_current.calls = i + 1;
  }

  int *stack_indices, *func_indices;
  int n_indices;
  vftr_find_function_in_table ("C", &func_indices, &n_indices, false);
  vftr_find_function_in_stack ("C", &stack_indices, &n_indices, false);
  stack_leaf_t *stack_tree = NULL;
  for (int i = 0;  i < n_indices; i++) {
  	// Why is n_functions_in_stack 2 instead of 3?
  	int n_functions_in_stack = 3;
  	int *stack_ids = (int*)malloc (n_functions_in_stack * sizeof(int));
  	int stack_id = stack_indices[i];
  	int function_id = func_indices[i];
  	for (int j = 0; j < n_functions_in_stack; j++) {
  		stack_ids[j] = stack_id;
  		stack_id = vftr_gStackinfo[stack_id].ret;
  		printf ("%d ", stack_id);
  	}
  	printf ("\n");
  	vftr_fill_into_stack_tree (&stack_tree, n_functions_in_stack, stack_ids, function_id);
  	free (stack_ids);
  }
  double dummy_d;
  int dummy_i;
  vftr_scan_stacktree (stack_tree->origin, 2, NULL, &dummy_d, &dummy_i, &dummy_d, &dummy_i, &dummy_i);
  display_function_t *display_functions[1];
  display_functions[0] = (display_function_t*)malloc (sizeof(display_function_t));
  display_functions[0]->func_name = "C";
  vftr_browse_print_stacktree_page (stdout, false, display_functions, 0, 1, stack_tree->origin, NULL, 0.0, 1000, 1);
  free (stack_tree);

#ifdef _MPI
  PMPI_Finalize();
#endif


  return 0;
}
