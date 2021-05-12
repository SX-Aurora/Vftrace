/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdbool.h>

#include "vftr_stringutils.h"
#include "vftr_setup.h"
#include "vftr_symbols.h"
#include "vftr_stacks.h"
#include "vftr_hashing.h"
#include "vftr_regex.h"
#include "vftr_environment.h"
#include "vftr_functions.h"
#include "vftr_fileutils.h"
#include "vftr_hwcounters.h"


char *vftr_precise_functions[] = {
   "MPI_Allgather", "MPI_Allgatherv", "MPI_Allreduce", "MPI_Alltoall",
   "MPI_Alltoallv", "MPI_Alltoallw", "MPI_Barrier", "MPI_Bcast",
   "MPI_Bsend", "MPI_Bsend_init", "MPI_Gather", "MPI_Gatherv",
   "MPI_Get", "MPI_Ibsend", "MPI_Irecv", "MPI_Irsend",
   "MPI_Isend", "MPI_Issend", "MPI_Put", "MPI_Recv", "MPI_Reduce",
   "MPI_Reduce_scatter", "MPI_Rsend", "MPI_Scatter", "MPI_Scatterv",
   "MPI_Send", "MPI_Sendrecv", "MPI_Sendrecv_replace", "MPI_Ssend",
   "MPI_Test", "MPI_Testall", "MPI_Testany", "MPI_Testsome",
   "MPI_Wait", "MPI_Waitall", "MPI_Waitany", "MPI_Waitsome",

   "MPI_Allgather_f08", "MPI_Allgatherv_f08", "MPI_Allreduce_f08", "MPI_Alltoall_f08",
   "MPI_Alltoallv_f08", "MPI_Alltoallw_f08", "MPI_Barrier_f08", "MPI_Bcast_f08",
   "MPI_Bsend_f08", "MPI_Bsend_init_f08", "MPI_Gather_f08", "MPI_Gatherv_f08",
   "MPI_Get_f08", "MPI_Ibsend_f08", "MPI_Irecv_f08", "MPI_Irsend_f08",
   "MPI_Isend_f08", "MPI_Issend_f08", "MPI_Put_f08", "MPI_Recv_f08", "MPI_Reduce_f08",
   "MPI_Reduce_scatter_f08", "MPI_Rsend_f08", "MPI_Scatter_f08", "MPI_Scatterv_f08",
   "MPI_Send_f08", "MPI_Sendrecv_f08", "MPI_Sendrecv_replace_f08", "MPI_Ssend_f08",
   "MPI_Test_f08", "MPI_Testall_f08", "MPI_Testany_f08", "MPI_Testsome_f08",
   "MPI_Wait_f08", "MPI_Waitall_f08", "MPI_Waitany_f08", "MPI_Waitsome_f08",

   "vftrace_pause", "vftrace_resume",
   "vftrace_get_stack",
   NULL // Null pointer to terminate the list
};

//int *vftr_print_stackid_list;
struct loc_glob_id *vftr_print_stackid_list;
int vftr_n_print_stackids;
int vftr_stackid_list_size;

void vftr_init_profdata (profdata_t *prof) {
  prof->calls = 0;
  prof->cycles = 0;
  prof->time_excl = 0;
  prof->time_incl = 0;
  prof->event_count = NULL; 
  prof->events[0] = NULL;
  prof->events[1] = NULL;
  prof->ic = 0;
  prof->mpi_tot_send_bytes = 0;
  prof->mpi_tot_recv_bytes = 0;
  
}

// add a new function to the stack tables
function_t *vftr_new_function(void *arg, const char *function_name, function_t *caller, bool is_precise) {

   // create and null new function
   function_t *func = (function_t *) malloc (sizeof(function_t));
   memset(func, 0, sizeof(function_t));

   // assign function's name
   if (function_name) {
      func->name = strdup(function_name);
   } else {
      char *symbol = vftr_find_symbol (arg);
      if (symbol) {
         func->name = strdup(symbol);
         /* Chop Fortran trailing underscore */
         int n = strlen(symbol);
         if ((symbol[n-1] == '_') && (symbol[n-2] != '_')) {
            func->name[n-1] = '\0';
         }
      } else {
         func->name = strdup("unknown");
      }
   }

   // Function address
   func->address = arg;
   // local unique stack ID
   func->id = vftr_stackscount;
   // global unique stack ID (unknown for now, so it gets an invalid value)
   func->gid = -1;
   // local unique stack ID of the calling function
   func->return_to = caller;
   // only for debugging
   func->new = true;
   func->detail = true;
   // if called recursively keep track of depth
   func->recursion_depth = 0;

   // compute the stack hash
   int stackstrlength = strlen(func->name);
   function_t *tmpfunc = func;
   // go down the stack until the bottom is reached
   // record the length of the function names each
   while (tmpfunc->return_to) {
      tmpfunc = tmpfunc->return_to;
      // add one chars for function division by "<"
      stackstrlength += 1;
      stackstrlength += strlen(tmpfunc->name);
   }

   // allocate space to hold the complete string
   char *stackstr = (char*) malloc((1+stackstrlength)*sizeof(char));
   char *strptr = stackstr;
   tmpfunc = func;
   // copy the first string in and move the strpointer forward
   strcpy(strptr, tmpfunc->name);
   strptr += strlen(tmpfunc->name);
   // go down the stack until the bottom is reached
   // copy the function names onto the string
   while (tmpfunc->return_to) {
      tmpfunc = tmpfunc->return_to;
      strcpy(strptr, "<");
      strptr += 1;
      strcpy(strptr, tmpfunc->name);
      strptr += strlen(tmpfunc->name);
   }
   // use the string to compute the individual callstack hash
   func->stackHash = vftr_jenkins_murmur_64_hash(stackstrlength, (uint8_t*) stackstr);

   // free the string;
   free(stackstr);

   if (arg) { // Skip if address not defined (when function is "init")
      func->precise = is_precise || vftr_pattern_match (vftr_environment.preciseregex->value, func->name);
   }

   // Check if the new function is meant to be pricisely sampled.
   // Linear search is fine as every function is only called once.
   char **precise_names = vftr_precise_functions;
   // move through the list until the terminating NULL pointer is reached
   while (*precise_names != NULL) {
      if (!strcmp(func->name, *precise_names)) {
         func->precise = true;
         break;
      }
      precise_names++;
   }

   // preparing the function specific profiling data
   vftr_init_profdata (&func->prof_current);
   vftr_init_profdata (&func->prof_previous);

   if (vftr_n_hw_obs > 0) {
      func->prof_current.event_count = (long long*) malloc(vftr_n_hw_obs * sizeof(long long));
      func->prof_previous.event_count = (long long*) malloc(vftr_n_hw_obs * sizeof(long long));
      memset (func->prof_current.event_count, 0, vftr_n_hw_obs * sizeof(long long));
      memset (func->prof_previous.event_count, 0, vftr_n_hw_obs * sizeof(long long));
   }

   // Determine if this function should be profiled
   func->profile_this = vftr_pattern_match(vftr_environment.runtime_profile_funcs->value, func->name);

   // Is this function a branch or the root of the calltree?
   if (caller != NULL) {
      if (caller->callee) {
         func->next_in_level = caller->callee->next_in_level;
      } else {
         caller->callee = caller->first_in_level = func;
      }
      caller->callee->next_in_level = func;
      caller->levels++;
   // Just copy the root from the caller
      func->root = caller->root;
   } else {
      func->root = func;
   }

   if (!vftr_func_table || (vftr_stackscount+1) > vftr_func_table_size) {
      // Allocate larger function table
      size_t newsize = 2*vftr_func_table_size;
      function_t **newtable = (function_t**) malloc(newsize * sizeof(function_t*));
      if (vftr_func_table) {
         memcpy(newtable, vftr_func_table,
                vftr_func_table_size * sizeof(function_t *));
         free(vftr_func_table);
      }
      vftr_func_table = newtable;
      vftr_func_table_size = newsize;
   }
   func->overhead = 0;
   vftr_func_table[vftr_stackscount++] = func;

   return func;
}

/**********************************************************************/

void vftr_reset_counts (function_t *func) {
   function_t *f;
   int i, n;
   int m = vftr_n_hw_obs * sizeof(long long);

   if (func == NULL) return;

   if (func->prof_current.event_count) {
     memset (func->prof_current.event_count,  0, m );
   }
   if (func->prof_previous.event_count) {
     memset (func->prof_previous.event_count, 0, m );
   }
   func->prof_current.cycles  = 0;
   func->prof_current.time_excl = 0;
   func->prof_current.time_incl = 0;
   n = func->levels;

   /* Recursive scan of callees */
   for (i = 0,f = func->first_in_level; i < n; i++, f = f->next_in_level) {
       vftr_reset_counts (f);
   }
}

/**********************************************************************/

void vftr_find_function_in_table (char *func_name, int **indices, int *n_indices, bool to_lower_case) {
	*n_indices = 0;
        bool equal;
	for (int i = 0; i < vftr_stackscount; i++) {
		if (to_lower_case) {
		   equal = !strcmp(vftr_to_lowercase(vftr_func_table[i]->name), vftr_to_lowercase(func_name));
		} else {
                   equal = !strcmp(vftr_func_table[i]->name, func_name);
                }
		if (equal) (*n_indices)++;
	}
	if (*n_indices > 0) {
		*indices = (int*)malloc(*n_indices * sizeof(int));
		int idx = 0;
		for (int i = 0; i < vftr_stackscount; i++) {
		   if (to_lower_case) {
         	      equal = !strcmp(vftr_to_lowercase(vftr_func_table[i]->name), vftr_to_lowercase(func_name));
                   } else {
                      equal = !strcmp(vftr_func_table[i]->name, func_name);
                   }
		   if (equal) (*indices)[idx++] = i;
		}
	}
}

/**********************************************************************/

void vftr_find_function_in_stack (char *func_name, int **indices, int *n_indices, bool to_lower_case) {
	*n_indices = 0;
	char *s_compare;
	for (int i = 0; i < vftr_gStackscount; i++) {
		s_compare = strdup (vftr_gStackinfo[i].name);
		if (to_lower_case) {
			for (int i = 0; i < strlen(s_compare); i++) {
				s_compare[i] = tolower(s_compare[i]);
			}
		}
		if (!strcmp (s_compare, func_name)) {
			(*n_indices)++;
		}
	}
	if (*n_indices > 0) {
		*indices = (int*)malloc(*n_indices * sizeof(int));
		int idx = 0;
		for (int i = 0; i < vftr_gStackscount; i++) {
		   s_compare = strdup (vftr_gStackinfo[i].name);
		   if (to_lower_case) {
		   	for (int i = 0; i < strlen(s_compare); i++) {
		   		s_compare[i] = tolower(s_compare[i]);
		   	}
		   }
		   if (!strcmp (s_compare, func_name)) {
		   	(*indices)[idx++] = i;
		   }
		}
	}
}

/**********************************************************************/

void vftr_write_function_indices (FILE *fp, char *func_name, bool to_lower_case) {
	int n_indices;
	int *func_indices = NULL;
	vftr_find_function_in_table (func_name, &func_indices, &n_indices, to_lower_case);
	if (!func_indices) {
		fprintf (fp, "ERROR: No indices found for function %s\n", func_name);
	} else {
		fprintf (fp, "%s found at indices: ", func_name);
		for (int i = 0; i < n_indices; i++) {
			fprintf (fp, "%d ", func_indices[i]);
		}
		fprintf (fp, "\n");	
		free (func_indices);
	}
}

/**********************************************************************/

void vftr_write_function (FILE *fp, function_t *func, bool verbose) {
	fprintf (fp, "Function: %s\n", func->name);
        if (!verbose) return;
	fprintf (fp, "\tAddress: ");
	if (func->address) {
		fprintf (fp, "%p\n", func->address);
	} else {
		fprintf (fp, "-/-\n");
	}
	fprintf (fp, "\tCalled from: ");
	if (func->return_to) {
		fprintf (fp, "%s\n", func->return_to->name);	
	} else {
		fprintf (fp, "-/-\n");
	}
	fprintf (fp, "\tCurrently calling: ");
	if (func->callee) {
		fprintf (fp, "%s\n", func->callee->name);
	} else {	
		fprintf (fp, "-/-\n");
	}
	fprintf (fp, "\tFirst in next Level: ");
	if (func->first_in_level) {
		fprintf (fp, "%s\n", func->first_in_level->name);
	} else {	
		fprintf (fp, "-/-\n");
	}
	fprintf (fp, "\tNext in current Level: ");
	if (func->next_in_level) {
		fprintf (fp, "%s\n", func->next_in_level->name);
	} else {	
		fprintf (fp, "-/-\n");
	}
	fprintf (fp, "\tRoot: ");
	if (func->root) {
		fprintf (fp, "%s\n", func->root->name);
	} else {	
		fprintf (fp, "-/-\n");
	}


	fprintf (fp, "\tprecise: %s\n", vftr_bool_to_string (func->precise));
	fprintf (fp, "\tID: %d\n", func->id);
	fprintf (fp, "\tGroup ID: %d\n", func->gid);
	fprintf (fp, "\tRecursion depth: %d\n", func->recursion_depth);
	fprintf (fp, "\tStackHash: %lu\n", func->stackHash);
}
		
/**********************************************************************/

void vftr_stackid_list_init () {
   vftr_n_print_stackids = 0;
   vftr_stackid_list_size = STACKID_LIST_INC;
   //vftr_print_stackid_list = (int*)malloc(STACKID_LIST_INC * sizeof(int));
   vftr_print_stackid_list = (struct loc_glob_id*) malloc (STACKID_LIST_INC * sizeof (struct loc_glob_id));
}

/**********************************************************************/

void vftr_stackid_list_add (int local_stack_id, int global_stack_id) {
   if (vftr_n_print_stackids + 1 > vftr_stackid_list_size) {
      vftr_stackid_list_size += STACKID_LIST_INC;
      //vftr_print_stackid_list = (int*)realloc (vftr_print_stackid_list, vftr_stackid_list_size * sizeof(int));
      vftr_print_stackid_list = (struct loc_glob_id*) realloc (vftr_print_stackid_list, vftr_stackid_list_size * sizeof (struct loc_glob_id));
   }
   //vftr_print_stackid_list[vftr_n_print_stackids++] = stack_id;
   vftr_print_stackid_list[vftr_n_print_stackids].loc = local_stack_id;
   vftr_print_stackid_list[vftr_n_print_stackids].glob = global_stack_id;
   vftr_n_print_stackids++;
}

/**********************************************************************/

void vftr_stackid_list_print (FILE *fp) {
   for (int i = 0; i < vftr_n_print_stackids; i++) {
      fprintf (fp, "(%d %d) ", vftr_print_stackid_list[i].loc, vftr_print_stackid_list[i].glob);
   }
   fprintf (fp, "\n");
}

/**********************************************************************/

void vftr_stackid_list_finalize () {
   free (vftr_print_stackid_list);
   vftr_stackid_list_size = 0;
   vftr_n_print_stackids = 0;
}

/**********************************************************************/

void vftr_strip_all_module_names () {
  for (int i = 0; i < vftr_stackscount; i++) {
     vftr_func_table[i]->name = vftr_strip_module_name (vftr_func_table[i]->name);
  }
}

/**********************************************************************/

#ifdef _LIBERTY_AVAIL
void vftr_demangle_all_func_names () {
  for (int i = 0; i < vftr_stackscount; i++) {
    vftr_func_table[i]->name = vftr_demangle_cpp (vftr_func_table[i]->name);
  }
}
#endif

/**********************************************************************/

int vftrace_show_stacktree_size () {
   return vftr_stackscount;
}

/**********************************************************************/

int vftr_functions_test_1 (FILE *fp_in, FILE *fp_out) {
	unsigned long long addr[1];
	fprintf (fp_out, "Initial vftr_stackscount: %d\n", vftr_stackscount);
	int i0 = vftr_stackscount;
	if (i0 > 0) fprintf (fp_out, "Check additional MPI entries:\n");
   	for (int i = 0; i < i0; i++) {
		vftr_write_function (fp_out, vftr_func_table[i], true);
	}
	function_t *func1 = vftr_new_function (NULL, "init_vftr", NULL, false);
	function_t *func2 = vftr_new_function ((void*)addr, "test_1", func1, true);
	fprintf (fp_out, "Check test entries:\n");
	for (int i = i0; i < vftr_stackscount; i++) {
		vftr_write_function (fp_out, vftr_func_table[i], true);
	}
	return 0;
}

/**********************************************************************/

int vftr_functions_test_2 (FILE *fp_in, FILE *fp_out) {
	fprintf (fp_out, "Initial vftr_stackscount: %d\n", vftr_stackscount);

	int i0 = vftr_stackscount;
	if (i0 > 0) fprintf (fp_out, "Check additional MPI entries:\n");
   	for (int i = 0; i < i0; i++) {
		vftr_write_function (fp_out, vftr_func_table[i], true);
	}

	unsigned long long addrs [6];
	function_t *func1 = vftr_new_function (NULL, "init_vftr", NULL, false);
	function_t *func2 = vftr_new_function ((void*)addrs, "func2", func1, false);
	function_t *func3 = vftr_new_function ((void*)(addrs + 1), "func3", func1, false);	
	function_t *func4 = vftr_new_function ((void*)(addrs + 2), "func4", func3, false);
	function_t *func5 = vftr_new_function ((void*)(addrs + 3), "func5", func2, false);
	function_t *func6 = vftr_new_function ((void*)(addrs + 4), "func6", func2, false);
	function_t *func7 = vftr_new_function ((void*)(addrs + 5), "func4", func6, false);
	fprintf (fp_out, "Check test entries:\n");
	for (int i = i0; i < vftr_stackscount; i++) {
		vftr_write_function(fp_out, vftr_func_table[i], true);
	}
	fprintf (fp_out, "Test if callee pointer is changed properly\n");
	func2->callee = func6;
	vftr_write_function (fp_out, func2, true);
	fprintf (fp_out, "vftr_func_table_size: %d\n", vftr_func_table_size);
	fprintf (fp_out, "vftr_stackscount: %d\n", vftr_stackscount);
	fprintf (fp_out, "Check functions registered in function table: \n");
	for (int i = i0; i < vftr_stackscount; i++) {
		vftr_write_function(fp_out, vftr_func_table[i], true);
	}
	return 0;
}

/**********************************************************************/

int vftr_functions_test_3 (FILE *fp_in, FILE *fp_out) {
	unsigned long long addrs [6];
	function_t *func1 = vftr_new_function (NULL, "init", NULL, false);
	function_t *func2 = vftr_new_function ((void*)addrs, "func2", func1, false);
	function_t *func3 = vftr_new_function ((void*)(addrs + 1), "func3", func1, false);	
	function_t *func4 = vftr_new_function ((void*)(addrs + 2), "func4", func3, false);
	function_t *func5 = vftr_new_function ((void*)(addrs + 3), "func5", func2, false);
	function_t *func6 = vftr_new_function ((void*)(addrs + 4), "func6", func2, false);
	function_t *func7 = vftr_new_function ((void*)(addrs + 5), "func4", func6, false);
	vftr_write_stack_ascii (fp_out, 0.0, func1, "", 0);
	vftr_write_stack_ascii (fp_out, 0.0, func2, "", 0);
	vftr_write_stack_ascii (fp_out, 0.0, func3, "", 0);
	vftr_write_stack_ascii (fp_out, 0.0, func4, "", 0);
	vftr_write_stack_ascii (fp_out, 0.0, func5, "", 0);
	vftr_write_stack_ascii (fp_out, 0.0, func6, "", 0);
	vftr_write_stack_ascii (fp_out, 0.0, func7, "", 0);
	return 0;
}

/**********************************************************************/

int vftr_functions_test_4 (FILE *fp_in, FILE *fp_out) {
	unsigned long long addrs [6];
	function_t *func1 = vftr_new_function (NULL, "init", NULL, false);
	function_t *func2 = vftr_new_function ((void*)addrs, "func2", func1, false);
	function_t *func3 = vftr_new_function ((void*)(addrs + 1), "func3", func1, false);	
	function_t *func4 = vftr_new_function ((void*)(addrs + 2), "func4", func3, false);
	function_t *func5 = vftr_new_function ((void*)(addrs + 3), "func2", func4, false);
	vftr_write_function_indices (fp_out, "init", false);
	vftr_write_function_indices (fp_out, "func2", false);
	vftr_write_function_indices (fp_out, "func3", false);
	vftr_write_function_indices (fp_out, "func4", false);
	return 0;
}

/**********************************************************************/

int vftr_functions_test_5 (FILE *fp_in, FILE *fp_out) {
	unsigned long long addrs [6];
	function_t *func1 = vftr_new_function (NULL, "INIT", NULL, false);
	function_t *func2 = vftr_new_function ((void*)addrs, "fUnC2", func1, false);
	function_t *func3 = vftr_new_function ((void*)(addrs + 1), "FUnc3", func1, false);	
	function_t *func4 = vftr_new_function ((void*)(addrs + 2), "func4", func3, false);
	function_t *func5 = vftr_new_function ((void*)(addrs + 3), "fUNC2", func4, false);
	vftr_write_function_indices (fp_out, "init", true);
	vftr_write_function_indices (fp_out, "func2", true);
	vftr_write_function_indices (fp_out, "func3", true);
	vftr_write_function_indices (fp_out, "func4", true);
	return 0;
}
