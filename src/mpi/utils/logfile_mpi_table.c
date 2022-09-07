#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "timer_types.h"
#include "table_types.h"
#include "environment_types.h"
#include "collated_stack_types.h"
#include "vftrace_state.h"
#include "mpiprofiling_types.h"

#include "filenames.h"
#include "license.h"
#include "config.h"
#include "environment.h"
#include "stacks.h"
#include "collate_stacks.h"
#include "tables.h"
//#include "sorting.h"

int *vftr_logfile_mpi_table_nmessages_list(int nstacks, collated_stack_t **stack_ptrs) {
   int *nmessages_list = (int*) malloc(nstacks*sizeof(int));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      nmessages_list[istack] = stack_ptr->profile.mpiProf.nsendmessages;
      nmessages_list[istack] += stack_ptr->profile.mpiProf.nrecvmessages;
   }
   return nmessages_list;
}

double *vftr_logfile_mpi_table_average_send_bytes_list(int nstacks, collated_stack_t **stack_ptrs) {
   double *avg_send_bytes_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      profile_t *prof_ptr = &(stack_ptr->profile);
      avg_send_bytes_list[istack] = prof_ptr->mpiProf.send_bytes;
      int nmessages = prof_ptr->mpiProf.nsendmessages;
      if (nmessages > 0) {
         avg_send_bytes_list[istack] /= nmessages;
      } else {
         avg_send_bytes_list[istack] = 0.0;
      }
   }
   return avg_send_bytes_list;
}

double *vftr_logfile_mpi_table_average_recv_bytes_list(int nstacks, collated_stack_t **stack_ptrs) {
   double *avg_recv_bytes_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      profile_t *prof_ptr = &(stack_ptr->profile);
      avg_recv_bytes_list[istack] = prof_ptr->mpiProf.recv_bytes;
      int nmessages = prof_ptr->mpiProf.nrecvmessages;
      if (nmessages > 0) {
         avg_recv_bytes_list[istack] /= nmessages;
      } else {
         avg_recv_bytes_list[istack] = 0.0;
      }
   }
   return avg_recv_bytes_list;
}

double *vftr_logfile_mpi_table_average_send_bw_list(int nstacks, collated_stack_t **stack_ptrs) {
   double *avg_send_bw_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      profile_t *prof_ptr = &(stack_ptr->profile);
      avg_send_bw_list[istack] = prof_ptr->mpiProf.acc_send_bw;
      int nmessages = prof_ptr->mpiProf.nsendmessages;
      if (nmessages > 0) {
         avg_send_bw_list[istack] /= nmessages;
      } else {
         avg_send_bw_list[istack] = 0.0;
      }
   }
   return avg_send_bw_list;
}

double *vftr_logfile_mpi_table_average_recv_bw_list(int nstacks, collated_stack_t **stack_ptrs) {
   double *avg_recv_bw_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      profile_t *prof_ptr = &(stack_ptr->profile);
      avg_recv_bw_list[istack] = prof_ptr->mpiProf.acc_recv_bw;
      int nmessages = prof_ptr->mpiProf.nrecvmessages;
      if (nmessages > 0) {
         avg_recv_bw_list[istack] /= nmessages;
      } else {
         avg_recv_bw_list[istack] = 0.0;
      }
   }
   return avg_recv_bw_list;
}

double *vftr_logfile_mpi_table_average_comm_time_list(int nstacks, collated_stack_t **stack_ptrs) {
   double *avg_comm_time_list = (double*) malloc(nstacks*sizeof(double));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      avg_comm_time_list[istack] = 0.0;
      profile_t *prof_ptr = &(stack_ptr->profile);
      long long tot_time = prof_ptr->mpiProf.total_time_usec;
      int nmessages = 0;
      nmessages += prof_ptr->mpiProf.nsendmessages;
      nmessages += prof_ptr->mpiProf.nrecvmessages;
      if (nmessages > 0) {
         avg_comm_time_list[istack] = tot_time * 1.0e-6/((double)nmessages);
      } else {
         avg_comm_time_list[istack] = 0.0;
      }
   }
   return avg_comm_time_list;
}

char **vftr_logfile_mpi_table_stack_function_name_list(int nstacks, collated_stack_t **stack_ptrs) {
   char **name_list = (char**) malloc(nstacks*sizeof(char*));
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      name_list[istack] = stack_ptr->name;
   }
   return name_list;
}

char **vftr_logfile_mpi_table_stack_caller_name_list(int nstacks,
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

char **vftr_logfile_mpi_table_callpath_list(int nstacks,
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

int *vftr_logfile_mpi_table_stack_globalstackID_list(int nstacks,
                                                     collated_stack_t **stack_ptrs) {
   int *id_list = (int*) malloc(nstacks*sizeof(int));
   int listidx = 0;
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stack_ptrs[istack];
      profile_t *prof_ptr = &(stack_ptr->profile);
      if (prof_ptr->mpiProf.nsendmessages > 0 ||
          prof_ptr->mpiProf.nrecvmessages > 0) {
         id_list[istack] = stack_ptr->gid;
         listidx++;
      }
   }
   return id_list;
}

int vftr_logfile_mpi_table_nrows(collated_stacktree_t stacktree) {
   int nstacks = stacktree.nstacks;
   int nrows = 0;
   for (int istack=0; istack<nstacks; istack++) {
      collated_stack_t *stack_ptr = stacktree.stacks+istack;
      mpiProfile_t mpiprof = stack_ptr->profile.mpiProf;
      if (mpiprof.nsendmessages > 0 ||
          mpiprof.nrecvmessages > 0) {
         nrows++;
      }
   }
   return nrows;
}

collated_stack_t **vftr_logfile_mpi_table_get_relevant_collated_stacks(
   int nrows, collated_stacktree_t stacktree) {

   collated_stack_t **selected_stacks = (collated_stack_t**)
      malloc(nrows*sizeof(collated_stack_t));
   int irow = 0;
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      collated_stack_t *stack_ptr = stacktree.stacks+istack;
      mpiProfile_t mpiprof = stack_ptr->profile.mpiProf;
      if (mpiprof.nsendmessages > 0 ||
          mpiprof.nrecvmessages > 0) {
         selected_stacks[irow] = stack_ptr;
         irow++;
      }
   }
   return selected_stacks;
}

void vftr_write_logfile_mpi_table(FILE *fp, collated_stacktree_t stacktree,
                                  environment_t environment) {
   SELF_PROFILE_START_FUNCTION;
   int nrows = vftr_logfile_mpi_table_nrows(stacktree);
   // TODO: sort something
   collated_stack_t **selected_stacks =
      vftr_logfile_mpi_table_get_relevant_collated_stacks(nrows, stacktree);

   fprintf(fp, "\nCommunication profile");
   if (environment.ranks_in_mpi_profile.set) {
      fprintf(fp, " for ranks: %s\n",
              environment.ranks_in_mpi_profile.value.string_val);
   } else {
      fprintf(fp, "\n");
   }

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, nrows);

   int *nmessages = vftr_logfile_mpi_table_nmessages_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_int, "Messages", "%d", 'c', 'r', (void*) nmessages);

   double *avg_send_bytes = vftr_logfile_mpi_table_average_send_bytes_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg send size/B", "%.3e", 'c', 'r', (void*) avg_send_bytes);

   double *avg_recv_bytes = vftr_logfile_mpi_table_average_recv_bytes_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg recv size/B", "%.3e", 'c', 'r', (void*) avg_recv_bytes);

   double *avg_send_bw = vftr_logfile_mpi_table_average_send_bw_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg send BW/B/s", "%.3le", 'c', 'r', (void*) avg_send_bw);

   double *avg_recv_bw = vftr_logfile_mpi_table_average_recv_bw_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg recv BW/B/s", "%.3le", 'c', 'r', (void*) avg_recv_bw);

   double *avg_comm_time = vftr_logfile_mpi_table_average_comm_time_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_double, "avg comm time/s", "%.3le", 'c', 'r', (void*) avg_comm_time);

   char **function_names = vftr_logfile_mpi_table_stack_function_name_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_string, "Function", "%s", 'c', 'r', (void*) function_names);

   char **caller_names = vftr_logfile_mpi_table_stack_caller_name_list(nrows, stacktree, selected_stacks);
   vftr_table_add_column(&table, col_string, "Caller", "%s", 'c', 'r', (void*) caller_names);

   int *stack_IDs = vftr_logfile_mpi_table_stack_globalstackID_list(nrows, selected_stacks);
   vftr_table_add_column(&table, col_int, "ID", "%d", 'c', 'r', (void*) stack_IDs);

   char **path_list=NULL;
   if (environment.callpath_in_mpi_profile.value.bool_val) {
      path_list = vftr_logfile_mpi_table_callpath_list(nrows,
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
   if (environment.callpath_in_profile.value.bool_val) {
      for (int irow=0; irow<nrows; irow++) {
         free(path_list[irow]);
      }
      free(path_list);
   }

   free(selected_stacks);
   SELF_PROFILE_END_FUNCTION;
}
