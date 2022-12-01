#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "timer_types.h"
#include "table_types.h"
#include "configuration_types.h"
#include "stack_types.h"
#include "vftrace_state.h"
#include "mpiprofiling_types.h"

#include "filenames.h"
#include "license.h"
#include "config.h"
#include "stacks.h"
#include "collate_stacks.h"
#include "tables.h"
#include "sorting.h"

int *vftr_ranklogfile_mpi_table_nmessages_list(int nstacks, vftr_stack_t **stack_ptrs) {
   int *nmessages_list = (int*) malloc(nstacks*sizeof(int));
   for (int istack=0; istack<nstacks; istack++) {
      vftr_stack_t *stack_ptr = stack_ptrs[istack];
      nmessages_list[istack] = 0;
      for (int iprof=0; iprof<stack_ptr->profiling.nprofiles; iprof++) {
         profile_t *prof_ptr = stack_ptr->profiling.profiles+iprof;
         nmessages_list[istack] += prof_ptr->mpiprof.nsendmessages;
         nmessages_list[istack] += prof_ptr->mpiprof.nrecvmessages;
      }
   }
   return nmessages_list;
}

double *vftr_ranklogfile_mpi_table_average_send_bytes_list(int nstacks, vftr_stack_t **stack_ptrs) {
   double *avg_send_bytes_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      vftr_stack_t *stack_ptr = stack_ptrs[istack];
      avg_send_bytes_list[istack] = 0.0;
      int nmessages = 0;
      for (int iprof=0; iprof<stack_ptr->profiling.nprofiles; iprof++) {
         profile_t *prof_ptr = stack_ptr->profiling.profiles+iprof;
         avg_send_bytes_list[istack] += prof_ptr->mpiprof.send_bytes;
         nmessages += prof_ptr->mpiprof.nsendmessages;
      }
      if (nmessages > 0) {
         avg_send_bytes_list[istack] /= nmessages;
      } else {
         avg_send_bytes_list[istack] = 0.0;
      }
   }
   return avg_send_bytes_list;
}

double *vftr_ranklogfile_mpi_table_average_recv_bytes_list(int nstacks, vftr_stack_t **stack_ptrs) {
   double *avg_recv_bytes_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      vftr_stack_t *stack_ptr = stack_ptrs[istack];
      avg_recv_bytes_list[istack] = 0.0;
      int nmessages = 0;
      for (int iprof=0; iprof<stack_ptr->profiling.nprofiles; iprof++) {
         profile_t *prof_ptr = stack_ptr->profiling.profiles+iprof;
         avg_recv_bytes_list[istack] += prof_ptr->mpiprof.recv_bytes;
         nmessages += prof_ptr->mpiprof.nrecvmessages;
      }
      if (nmessages > 0) {
         avg_recv_bytes_list[istack] /= nmessages;
      } else {
         avg_recv_bytes_list[istack] = 0.0;
      }
   }
   return avg_recv_bytes_list;
}

double *vftr_ranklogfile_mpi_table_average_send_bw_list(int nstacks, vftr_stack_t **stack_ptrs) {
   double *avg_send_bw_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      vftr_stack_t *stack_ptr = stack_ptrs[istack];
      avg_send_bw_list[istack] = 0.0;
      int nmessages = 0;
      for (int iprof=0; iprof<stack_ptr->profiling.nprofiles; iprof++) {
         profile_t *prof_ptr = stack_ptr->profiling.profiles+iprof;
         avg_send_bw_list[istack] += prof_ptr->mpiprof.acc_send_bw;
         nmessages += prof_ptr->mpiprof.nsendmessages;
      }
      if (nmessages > 0) {
         avg_send_bw_list[istack] /= nmessages;
      } else {
         avg_send_bw_list[istack] = 0.0;
      }
   }
   return avg_send_bw_list;
}

double *vftr_ranklogfile_mpi_table_average_recv_bw_list(int nstacks, vftr_stack_t **stack_ptrs) {
   double *avg_recv_bw_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      vftr_stack_t *stack_ptr = stack_ptrs[istack];
      avg_recv_bw_list[istack] = 0.0;
      int nmessages = 0;
      for (int iprof=0; iprof<stack_ptr->profiling.nprofiles; iprof++) {
         profile_t *prof_ptr = stack_ptr->profiling.profiles+iprof;
         avg_recv_bw_list[istack] += prof_ptr->mpiprof.acc_recv_bw;
         nmessages += prof_ptr->mpiprof.nrecvmessages;
      }
      if (nmessages > 0) {
         avg_recv_bw_list[istack] /= nmessages;
      } else {
         avg_recv_bw_list[istack] = 0.0;
      }
   }
   return avg_recv_bw_list;
}

double *vftr_ranklogfile_mpi_table_average_comm_time_list(int nstacks, vftr_stack_t **stack_ptrs) {
   double *avg_comm_time_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      vftr_stack_t *stack_ptr = stack_ptrs[istack];
      avg_comm_time_list[istack] = 0.0;
      int nmessages = 0;
      long long tot_time = 0;
      for (int iprof=0; iprof<stack_ptr->profiling.nprofiles; iprof++) {
         profile_t *prof_ptr = stack_ptr->profiling.profiles+iprof;
         tot_time += prof_ptr->mpiprof.total_time_nsec;
         nmessages += prof_ptr->mpiprof.nsendmessages;
         nmessages += prof_ptr->mpiprof.nrecvmessages;
      }
      if (nmessages > 0) {
         avg_comm_time_list[istack] = tot_time * 1.0e-9/((double)nmessages);
      } else {
         avg_comm_time_list[istack] = 0.0;
      }
   }
   return avg_comm_time_list;
}

char **vftr_ranklogfile_mpi_table_stack_function_name_list(int nstacks, vftr_stack_t **stack_ptrs) {
   char **name_list = (char**) malloc(nstacks*sizeof(char*));
   for (int istack=0; istack<nstacks; istack++) {
      vftr_stack_t *stack_ptr = stack_ptrs[istack];
      name_list[istack] = stack_ptr->name;
   }
   return name_list;
}

char **vftr_ranklogfile_mpi_table_stack_caller_name_list(int nstacks, stacktree_t stacktree,
                                                     vftr_stack_t **stack_ptrs) {
   char **name_list = (char**) malloc(nstacks*sizeof(char*));
   for (int istack=0; istack<nstacks; istack++) {
      vftr_stack_t *stack_ptr = stack_ptrs[istack];
      int callerID = stack_ptr->caller;
      if (callerID >= 0) {
         name_list[istack] = stacktree.stacks[callerID].name;
      } else {
         name_list[istack] = "----";
      }
   }
   return name_list;
}

char **vftr_ranklogfile_mpi_table_callpath_list(int nstacks, vftr_stack_t **stack_ptrs,
                                            stacktree_t stacktree) {
   char **path_list = (char**) malloc(nstacks*sizeof(char*));
   for (int istack=0; istack<nstacks; istack++) {
      vftr_stack_t *stack_ptr = stack_ptrs[istack];
      int stackid = stack_ptr->lid;
      path_list[istack] = vftr_get_stack_string(stacktree, stackid, false);
   }
   return path_list;
}

int *vftr_ranklogfile_mpi_table_stack_globalstackID_list(int nstacks, vftr_stack_t **stack_ptrs) {
   int *id_list = (int*) malloc(nstacks*sizeof(int));
   int listidx = 0;
   for (int istack=0; istack<nstacks; istack++) {
      vftr_stack_t *stack_ptr = stack_ptrs[istack];
      for (int iprof=0; iprof<stack_ptr->profiling.nprofiles; iprof++) {
         profile_t *prof_ptr = stack_ptr->profiling.profiles+iprof;
         if (prof_ptr->mpiprof.nsendmessages > 0 ||
             prof_ptr->mpiprof.nrecvmessages > 0) {
            id_list[istack] = stack_ptr->gid;
            listidx++;
            break;
         }
      }
   }
   return id_list;
}

int vftr_ranklogfile_mpi_table_nrows(stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   int nrows = 0;
   for (int istack=0; istack<nstacks; istack++) {
      vftr_stack_t *stack_ptr = stacktree.stacks+istack;
      for (int iprof=0; iprof<stack_ptr->profiling.nprofiles; iprof++) {
         profile_t *prof_ptr = stack_ptr->profiling.profiles+iprof;
         if (prof_ptr->mpiprof.nsendmessages > 0 ||
             prof_ptr->mpiprof.nrecvmessages > 0) {
            nrows++;
            break;
         }
      }
   }
   return nrows;
}

vftr_stack_t **vftr_ranklogfile_mpi_table_get_relevant_stacks(int nrows,
                                                     stacktree_t stacktree) {
   vftr_stack_t **selected_stacks = (vftr_stack_t**) malloc(nrows*sizeof(vftr_stack_t));
   int irow = 0;
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      vftr_stack_t *stack_ptr = stacktree.stacks+istack;
      for (int iprof=0; iprof<stack_ptr->profiling.nprofiles; iprof++) {
         profile_t *prof_ptr = stack_ptr->profiling.profiles+iprof;
         if (prof_ptr->mpiprof.nsendmessages > 0 ||
             prof_ptr->mpiprof.nrecvmessages > 0) {
            selected_stacks[irow] = stack_ptr;
            irow++;
            break;
         }
      }
   }
   return selected_stacks;
}

void vftr_write_ranklogfile_mpi_table(FILE *fp, stacktree_t stacktree,
                                      config_t config) {
   SELF_PROFILE_START_FUNCTION;
   int nrows = vftr_ranklogfile_mpi_table_nrows(stacktree);
   vftr_stack_t **selected_stacks =
      vftr_ranklogfile_mpi_table_get_relevant_stacks(nrows, stacktree);
   vftr_sort_stacks_for_mpiprof(config, nrows, selected_stacks);

   fprintf(fp, "\nCommunication profile");
   if (config.mpi.only_for_ranks.set) {
      fprintf(fp, " for ranks: %s\n",
              config.mpi.only_for_ranks.value);
   } else {
      fprintf(fp, "\n");
   }

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, nrows);

   int *nmessages = vftr_ranklogfile_mpi_table_nmessages_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_int, "Messages", "%d", 'c', 'r', (void*) nmessages);

   double *avg_send_bytes = vftr_ranklogfile_mpi_table_average_send_bytes_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg send size[B]", "%.3e", 'c', 'r', (void*) avg_send_bytes);

   double *avg_recv_bytes = vftr_ranklogfile_mpi_table_average_recv_bytes_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg recv size[B]", "%.3e", 'c', 'r', (void*) avg_recv_bytes);

   double *avg_send_bw = vftr_ranklogfile_mpi_table_average_send_bw_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg send BW[B/s]", "%.3le", 'c', 'r', (void*) avg_send_bw);

   double *avg_recv_bw = vftr_ranklogfile_mpi_table_average_recv_bw_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg recv BW[B/s]", "%.3le", 'c', 'r', (void*) avg_recv_bw);

   double *avg_comm_time = vftr_ranklogfile_mpi_table_average_comm_time_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg comm time[s]", "%.3le", 'c', 'r', (void*) avg_comm_time);

   char **function_names = vftr_ranklogfile_mpi_table_stack_function_name_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_string, "Function", "%s", 'c', 'r', (void*) function_names);

   char **caller_names = vftr_ranklogfile_mpi_table_stack_caller_name_list(nrows, stacktree, selected_stacks);
   vftr_table_add_column(&table, col_string, "Caller", "%s", 'c', 'r', (void*) caller_names);

   int *stack_IDs = vftr_ranklogfile_mpi_table_stack_globalstackID_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_int, "STID", "%d", 'c', 'r', (void*) stack_IDs);

   char **path_list=NULL;
   if (config.mpi.show_callpath.value) {
      path_list = vftr_ranklogfile_mpi_table_callpath_list(nrows,
                                                       selected_stacks,
                                                       stacktree);
      vftr_table_add_column(&table, col_string, "Callpath", "%s", 'c', 'r', (void*) path_list);
   }

   vftr_print_table(fp, table);

   vftr_table_free(&table);
   free(nmessages);
   free(avg_send_bytes);
   free(avg_recv_bytes);
   free(avg_send_bw);
   free(avg_recv_bw);
   free(avg_comm_time);
   free(function_names);
   free(caller_names);
   free(stack_IDs);
   if (config.mpi.show_callpath.value) {
      for (int irow=0; irow<nrows; irow++) {
         free(path_list[irow]);
      }
      free(path_list);
   }

   free(selected_stacks);
   SELF_PROFILE_END_FUNCTION;
}
