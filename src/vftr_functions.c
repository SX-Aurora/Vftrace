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
#include <stdbool.h>

#include "vftr_symbols.h"
#include "vftr_stacks.h"
#include "vftr_hashing.h"
#include "vftr_regex.h"
#include "vftr_environment.h"
#include "vftr_functions.h"
#include "vftr_hwcounters.h"


char *vftr_precice_functions[] = {
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
   "MPI_Wait", "MPI_Waitall", "MPI_Waitany", "MPI_Waitsome",

   "vftrace_pause", "vftrace_resume",
   "vftrace_get_stack",
   NULL // Null pointer to terminate the list
};

// add a new function to the stack tables
function_t *vftr_new_function(void *arg, const char *function_name,
                              function_t *caller, char *info, int line,
                              bool isPrecise) {

   // create and null new function
   function_t *func = (function_t *) malloc (sizeof(function_t));
   memset(func, 0, sizeof(function_t));

   // assign function's name
   if (function_name) {
      func->name = strdup(function_name);
   } else {
      char *symbol = vftr_find_symbol (arg, line, &(func->full));
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
   func->ret = caller;
   // only for debugging
   func->new = 1;
   func->detail = 1;
   // if called recursively keep track of depth
   func->recursion_depth = 0;

   // compute the stack hash
   int stackstrlength = strlen(func->name);
   function_t *tmpfunc = func;
   // go down the stack until the bottom is reached
   // record the length of the function names each
   while (tmpfunc->ret) {
      tmpfunc = tmpfunc->ret;
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
   while (tmpfunc->ret) {
      tmpfunc = tmpfunc->ret;
      strcpy(strptr, "<");
      strptr += 1;
      strcpy(strptr, tmpfunc->name);
      strptr += strlen(tmpfunc->name);
   }
   // use the string to compute the individual callstack hash
   func->stackHash = vftr_jenkins_murmur_64_hash(stackstrlength, (uint8_t*) stackstr);

   // free the string;
   free(stackstr);

   if (line > 0) func->line_beg = line;

   if (arg) { // Skip if address not defined (when info is "init")
      func->openmp = vftr_pattern_match(vftr_openmpregexp, func->name);
      func->precise = isPrecise || func->openmp ||
                      vftr_pattern_match(vftr_environment->preciseregex->value,
                                         func->name);
   }

   // Check if the new function is meant to be priceicely sampled
   // linear search is fine as every function is only called once
   char **precice_names = vftr_precice_functions;
   // move through the list until the terminating NULL pointer is reached
   while (*precice_names != NULL) {
      if (!strcmp(func->name, *precice_names)) {
         func->precise = true;
         break;
      }
      precice_names++;
   }

   // preparing the function specific profiling data
   int n = vftr_omp_threads * sizeof(profdata_t);
   func->prof_current = (profdata_t *) malloc(n);
   func->prof_previous = (profdata_t *) malloc(n);
   memset(func->prof_current, 0, n);
   memset(func->prof_previous, 0, n);

   if (vftr_n_hw_obs > 0) {
   	for (int i = 0; i < vftr_omp_threads; i++) {
   	   func->prof_current[i].event_count = (long long*) malloc(vftr_n_hw_obs * sizeof(long long));
   	   func->prof_previous[i].event_count = (long long*) malloc(vftr_n_hw_obs * sizeof(long long));
   	   memset (func->prof_current[i].event_count, 0, vftr_n_hw_obs * sizeof(long long));
   	   memset (func->prof_previous[i].event_count, 0, vftr_n_hw_obs * sizeof(long long));
   	}
   }

   // Determine if this function should be profiled
   func->profile_this = vftr_pattern_match(vftr_environment->runtime_profile_funcs->value, func->name);

   if (!func->exclude_this) {
      if (vftr_environment->include_only_regex->set) {
         func->exclude_this = !vftr_pattern_match(vftr_environment->include_only_regex->value, func->name);
      } else if (vftr_environment->exclude_functions_regex->set) {
         func->exclude_this = vftr_pattern_match(vftr_environment->exclude_functions_regex->value, func->name);
      }
   }

   // Is this function a branch or the root of the calltree?
   if (caller != NULL) {
      if (caller->call) {
         func->next   = caller->call->next;
      } else {
         caller->call = caller->first = func;
      }
      caller->call->next = func;
      caller->levels++;
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
   vftr_func_table[vftr_stackscount++] = func;

   return func;
}

void vftr_reset_counts (int me, function_t *func) {
   function_t *f;
   int i, n;
   int m = vftr_n_hw_obs * sizeof(long long);

   if( func == NULL ) return;

   memset (func->prof_current[me].event_count,  0, m );
   memset (func->prof_previous[me].event_count, 0, m );
   func->prof_current[me].calls   = 0;
   func->prof_current[me].cycles  = 0;
   func->prof_current[me].timeExcl = 0;
   func->prof_current[me].timeIncl = 0;
   func->prof_current[me].flops   = 0;
   n = func->levels;

   /* Recursive scan of callees */
   for (i = 0,f = func->first; i < n; i++, f = f->next) {
       vftr_reset_counts (me, f);
   }
}

