#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "self_profile.h"
#include "timer_types.h"
#include "table_types.h"
#include "configuration_types.h"
#include "collated_stack_types.h"
#include "vftrace_state.h"
#include "vedaprofiling_types.h"

#include "config.h"
#include "stacks.h"
#include "collate_stacks.h"
#include "tables.h"
#include "sorting.h"

int *vftr_logfile_veda_table_ncalls_list(int nstacks, collated_stack_t **stack_ptrs) {
   int *ncalls_list = (int*) malloc(nstacks*sizeof(int));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      ncalls_list[istack] = stack_ptr->profile.vedaprof.ncalls;
   }
   return ncalls_list;
}

double *vftr_logfile_veda_table_average_memcpy_HtoD_list(
   int nstacks, collated_stack_t **stack_ptrs) {
   double *memcpy_HtoD_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      memcpy_HtoD_list[istack] = stack_ptr->profile.vedaprof.HtoD_bytes;
      int ncalls = stack_ptr->profile.vedaprof.ncalls;
      if (ncalls > 0) {
         memcpy_HtoD_list[istack] /= ncalls;
      } else {
         memcpy_HtoD_list[istack] = 0.0;
      }
   }
   return memcpy_HtoD_list;
}

double *vftr_logfile_veda_table_average_memcpy_DtoH_list(
   int nstacks, collated_stack_t **stack_ptrs) {
   double *memcpy_DtoH_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      memcpy_DtoH_list[istack] = stack_ptr->profile.vedaprof.DtoH_bytes;
      int ncalls = stack_ptr->profile.vedaprof.ncalls;
      if (ncalls > 0) {
         memcpy_DtoH_list[istack] /= ncalls;
      } else {
         memcpy_DtoH_list[istack] = 0.0;
      }
   }
   return memcpy_DtoH_list;
}

double *vftr_logfile_veda_table_average_memcpy_HtoD_bw_list(
   int nstacks, collated_stack_t **stack_ptrs) {
   double *memcpy_HtoD_bw_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      memcpy_HtoD_bw_list[istack] = stack_ptr->profile.vedaprof.acc_HtoD_bw;
      int ncalls = stack_ptr->profile.vedaprof.ncalls;
      if (ncalls > 0) {
         memcpy_HtoD_bw_list[istack] /= ncalls;
      } else {
         memcpy_HtoD_bw_list[istack] = 0.0;
      }
   }
   return memcpy_HtoD_bw_list;
}

double *vftr_logfile_veda_table_average_memcpy_DtoH_bw_list(
   int nstacks, collated_stack_t **stack_ptrs) {
   double *memcpy_DtoH_bw_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      memcpy_DtoH_bw_list[istack] = stack_ptr->profile.vedaprof.acc_DtoH_bw;
      int ncalls = stack_ptr->profile.vedaprof.ncalls;
      if (ncalls > 0) {
         memcpy_DtoH_bw_list[istack] /= ncalls;
      } else {
         memcpy_DtoH_bw_list[istack] = 0.0;
      }
   }
   return memcpy_DtoH_bw_list;
}

double *vftr_logfile_veda_table_avg_time_list(int nstacks, collated_stack_t **stack_ptrs) {
   double *avg_time_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      avg_time_list[istack] = stack_ptr->profile.vedaprof.total_time_nsec*1.0e-9;
   }
   return avg_time_list;
}

char **vftr_logfile_veda_table_stack_function_name_list(int nstacks, collated_stack_t **stack_ptrs) {
   char **name_list = (char**) malloc(nstacks*sizeof(char*));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      name_list[istack] = stack_ptr->name;
   }
   return name_list;
}

char **vftr_logfile_veda_table_stack_caller_name_list(int nstacks,
                                                      collated_stacktree_t stacktree,
                                                      collated_stack_t **stack_ptrs) {
   char **name_list = (char**) malloc(nstacks*sizeof(char*));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      int callerID = stack_ptr->caller;
      if (callerID >= 0) {
         name_list[istack] = stacktree.stacks[callerID].name;
      } else {
         name_list[istack] = "----";
      }
   }
   return name_list;
}

char **vftr_logfile_veda_table_callpath_list(int nstacks,
                                            collated_stack_t **stack_ptrs,
                                            collated_stacktree_t stacktree) {
   char **path_list = (char**) malloc(nstacks*sizeof(char*));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      int stackid = stack_ptr->gid;
      path_list[istack] = vftr_get_collated_stack_string(stacktree, stackid, false);
   }
   return path_list;
}

int *vftr_logfile_veda_table_stack_globalstackID_list(int nstacks,
                                                     collated_stack_t **stack_ptrs) {
   int *id_list = (int*) malloc(nstacks*sizeof(int));
   int listidx = 0;
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      collated_profile_t *prof_ptr = &(stack_ptr->profile);
      if (prof_ptr->vedaprof.ncalls) {
         id_list[istack] = stack_ptr->gid;
         listidx++;
      }
   }
   return id_list;
}

int vftr_logfile_veda_table_nrows(collated_stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   int nrows = 0;
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stacktree.stacks+istack;
      vedaprofile_t vedaprof = stack_ptr->profile.vedaprof;
      if (vedaprof.ncalls > 0) {
         nrows++;
      }
   }
   return nrows;
}

collated_stack_t **vftr_logfile_veda_table_get_relevant_collated_stacks(
   int nrows, collated_stacktree_t stacktree) {

   collated_stack_t **selected_stacks = (collated_stack_t**)
      malloc(nrows*sizeof(collated_stack_t));
   int irow = 0;
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      collated_stack_t *stack_ptr = stacktree.stacks+istack;
      vedaprofile_t vedaprof = stack_ptr->profile.vedaprof;
      if (vedaprof.ncalls > 0) {
         selected_stacks[irow] = stack_ptr;
         irow++;
      }
   }
   return selected_stacks;
}

void vftr_write_logfile_veda_table(FILE *fp, collated_stacktree_t stacktree,
                                   config_t config) {
   SELF_PROFILE_START_FUNCTION;
   int nrows = vftr_logfile_veda_table_nrows(stacktree);
   collated_stack_t **selected_stacks =
      vftr_logfile_veda_table_get_relevant_collated_stacks(nrows, stacktree);
   //TODO vftr_sort_collated_stacks_for_vedaprof(config, nrows, selected_stacks);

   fprintf(fp, "\nVEDA profile\n");
   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, nrows);

   int *ncalls = vftr_logfile_veda_table_ncalls_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_int, "ncalls", "%d", 'c', 'r', (void*) ncalls);

   double *avg_memcpy_HtoD_list = vftr_logfile_veda_table_average_memcpy_HtoD_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg cpy H2D[B]", "%.3e", 'c', 'r', (void*) avg_memcpy_HtoD_list);
   
   double *avg_memcpy_DtoH_list = vftr_logfile_veda_table_average_memcpy_DtoH_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg cpy D2H[B]", "%.3e", 'c', 'r', (void*) avg_memcpy_DtoH_list);

   double *avg_memcpy_HtoD_bw_list = vftr_logfile_veda_table_average_memcpy_HtoD_bw_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg cpy H2D BW[B/s]", "%.3e", 'c', 'r', (void*) avg_memcpy_HtoD_bw_list);
   
   double *avg_memcpy_DtoH_bw_list = vftr_logfile_veda_table_average_memcpy_DtoH_bw_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg cpy D2H BW[B/s]", "%.3e", 'c', 'r', (void*) avg_memcpy_DtoH_bw_list);

   double *avg_time_list = vftr_logfile_veda_table_avg_time_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg time[s]", "%.3e", 'c', 'r', (void*) avg_time_list);

   char **function_names = vftr_logfile_veda_table_stack_function_name_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_string, "Function", "%s", 'c', 'r', (void*) function_names);

   char **caller_names = vftr_logfile_veda_table_stack_caller_name_list(nrows, stacktree, selected_stacks);
   vftr_table_add_column(&table, col_string, "Caller", "%s", 'c', 'r', (void*) caller_names);

   int *stack_IDs = vftr_logfile_veda_table_stack_globalstackID_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_int, "STID", "%d", 'c', 'r', (void*) stack_IDs);

   char **path_list=NULL;
//   if (config.veda.show_callpath.value) {
//      path_list = vftr_logfile_veda_table_callpath_list(nrows,
//                                                       selected_stacks,
//                                                       stacktree);
//      vftr_table_add_column(&table, col_string, "Callpath", "%s", 'c', 'r', (void*) path_list);
//   }


   vftr_print_table(fp, table);
   vftr_table_free(&table);
   free(ncalls);
   free(avg_memcpy_HtoD_list);
   free(avg_memcpy_DtoH_list);
   free(avg_memcpy_HtoD_bw_list);
   free(avg_memcpy_DtoH_bw_list);
   free(avg_time_list);
   free(function_names);
   free(caller_names);
   free(stack_IDs);
//TODO   if (config.veda.show_callpath.value) {
//TODO      for (int irow=0; irow<nrows; irow++) {
//TODO         free(path_list[irow]);
//TODO      }
//TODO      free(path_list);
//TODO   }

   free(selected_stacks);
   SELF_PROFILE_END_FUNCTION;
} 
