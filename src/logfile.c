#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "timer_types.h"
#include "table_types.h"
#include "environment_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "license.h"
#include "config.h"
#include "log_profile.h"
#include "environment.h"
#include "stacks.h"
#include "collate_stacks.h"
#include "tables.h"
#include "overheadprofiling_types.h"
#include "overheadprofiling.h"

char *vftr_get_logfile_name(environment_t environment, int rankID, int nranks) {
   char *filename_base = vftr_create_filename_base(environment, rankID, nranks);
   int filename_base_len = strlen(filename_base);

   char *extension = ".log";
   int extension_len = strlen(extension);

   // construct logfile name
   int total_len = filename_base_len +
                   extension_len +
                   1; // null terminator
   char *logfile_name = (char*) malloc(total_len*sizeof(char));
   strcpy(logfile_name, filename_base);
   strcat(logfile_name, extension);

   free(filename_base);
   return logfile_name;
}

void vftr_write_logfile_header(FILE *fp, time_strings_t timestrings,
                               environment_t environment) {
   fprintf(fp, "%s\n", PACKAGE_STRING);
   fprintf(fp, "Runtime profile for application:\n");
   fprintf(fp, "Start Date: %s\n", timestrings.start_time);
   fprintf(fp, "End Date:   %s\n\n", timestrings.end_time);
   // print the full license if requested by the environment
   if (environment.license_verbose.value.bool_val) {
      vftr_print_licence(fp);
   } else {
      vftr_print_licence_short(fp, environment.license_verbose.name);
   }
}

void vftr_write_logfile_summary(FILE *fp, process_t process, long long runtime) {
   double runtime_sec = runtime * 1.0e-6;

   // get the different accumulated overheads
   // The application runtime is the runtime minus the
   // sum of all overheads on the master thread
   long long total_master_overhead = 0ll;
   int nthreads = process.threadtree.nthreads;
   long long *hook_overheads = vftr_get_total_hook_overhead(process.stacktree, nthreads);
#ifdef _MPI
   long long *mpi_overheads = vftr_get_total_mpi_overhead(process.stacktree, nthreads);
#endif
#ifdef _OMP
   long long *omp_overheads = vftr_get_total_omp_overhead(process.stacktree, nthreads);
#endif
   for (int ithread=0; ithread<nthreads; ithread++) {
      if (process.threadtree.threads[ithread].master) {
         total_master_overhead += hook_overheads[ithread];
#ifdef _MPI
         total_master_overhead += mpi_overheads[ithread];
#endif
#ifdef _OMP
         total_master_overhead += omp_overheads[ithread];
#endif
      }
   }
   double total_master_overhead_sec = total_master_overhead*1.0e-6;
   double apptime_sec = runtime_sec - total_master_overhead_sec;

   fprintf(fp, "\n");
#ifdef _MPI
   fprintf(fp, "Nr. of MPI ranks:     %8d\n", process.nprocesses);
#endif
#ifdef _OMP
   fprintf(fp, "Nr. of OMP threads:   %8d\n", nthreads);
#endif
   fprintf(fp, "Total runtime:        %8.2lf s\n", runtime_sec);
   fprintf(fp, "Application time:     %8.2lf s\n", apptime_sec);
   fprintf(fp, "Overhead:             %8.2lf s\n", total_master_overhead_sec);
   if (nthreads == 1) {
      fprintf(fp, "   Function hooks:    %8.2lf s\n", hook_overheads[0]*1.0e-6);
#ifdef _MPI
      fprintf(fp, "   MPI wrappers:      %8.2lf s\n", mpi_overheads[0]*1.0e-6);
#endif
#ifdef _OMP
      fprintf(fp, "   OMP callbacks:     %8.2lf s\n", omp_overheads[0]*1.0e-6);
#endif
   } else {
      fprintf(fp, "   Function hooks:\n");
      for (int ithread=0; ithread<nthreads; ithread++) {
         fprintf(fp, "      Thread %d:      %8.2lf s\n",
                 ithread, hook_overheads[ithread]*1.0e-6);
      }
#ifdef _MPI
      fprintf(fp, "   MPI wrappers:\n");
      for (int ithread=0; ithread<nthreads; ithread++) {
         fprintf(fp, "      Thread %d:      %8.2lf s\n",
                 ithread, mpi_overheads[ithread]*1.0e-6);
      }
#endif
#ifdef _OMP
      fprintf(fp, "   OMP callbacks:\n");
      for (int ithread=0; ithread<nthreads; ithread++) {
         fprintf(fp, "      Thread %d:      %8.2lf s\n",
                 ithread, omp_overheads[ithread]*1.0e-6);
      }
#endif
   }
}

void vftr_write_logfile_profile_table(FILE *fp, stacktree_t stacktree,
                                      environment_t environment,
                                      long long runtime) {
   fprintf(fp, "\nRuntime profile\n");

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, stacktree.nstacks);

   int *calls = vftr_stack_calls_list(stacktree);
   vftr_table_add_column(&table, col_int, "Calls", "%d", 'c', 'r', (void*) calls);

   double *excl_time = vftr_stack_exclusive_time_list(stacktree);
   vftr_table_add_column(&table, col_double, "t_excl/s", "%.3f", 'c', 'r', (void*) excl_time);

   double *incl_time = vftr_stack_inclusive_time_list(stacktree);
   vftr_table_add_column(&table, col_double, "t_incl/s", "%.3f", 'c', 'r', (void*) incl_time);

  //
  // double *vftr_stack_overhead_time_list(int nstacks, stack_t *stacks);

   char **function_names = vftr_stack_function_name_list(stacktree);
   vftr_table_add_column(&table, col_string, "Function", "%s", 'c', 'r', (void*) function_names);

   char **caller_names = vftr_stack_caller_name_list(stacktree);
   vftr_table_add_column(&table, col_string, "Caller", "%s", 'c', 'r', (void*) caller_names);

   vftr_print_table(fp, table);

   vftr_table_free(&table);
   free(calls);
   free(excl_time);
   free(incl_time);
   free(function_names);
   free(caller_names);
}

void vftr_write_logfile_global_stack_list(FILE *fp, collated_stacktree_t stacktree) {
   fprintf(fp, "\nGlobal call stacks:\n");

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, stacktree.nstacks);
   vftr_table_left_outline(&table, false);
   vftr_table_right_outline(&table, false);
   vftr_table_columns_separating_line(&table, false);

   // first column with the StackIDs
   int *IDs = (int*) malloc(stacktree.nstacks*sizeof(int));
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      IDs[istack] = istack;
   }
   vftr_table_add_column(&table, col_int, "ID", "STID%d", 'r', 'r', (void*) IDs);

   // second column with the stack strings
   char **stacks = (char**) malloc(stacktree.nstacks*sizeof(char*));
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      stacks[istack] = vftr_get_collated_stack_string(stacktree, istack);
   }
   vftr_table_add_column(&table, col_string,
                         "Call stack", "%s", 'r', 'l', (void*) stacks);

   vftr_print_table(fp, table);

   vftr_table_free(&table);
   free(IDs);
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      free(stacks[istack]);
   }
   free(stacks);

}

FILE *vftr_open_logfile(char *filename) {
   FILE *fp = fopen(filename, "w");
   if (fp == NULL) {
      perror(filename);
      abort();
   }
   return fp;
}

void vftr_write_logfile(vftrace_t vftrace, long long runtime) {
   char *logfilename = vftr_get_logfile_name(vftrace.environment,
                                             vftrace.process.processID,
                                             vftrace.process.nprocesses);
   FILE *fp = vftr_open_logfile(logfilename);

   vftr_write_logfile_header(fp, vftrace.timestrings,
                             vftrace.environment);

   // print environment info
   vftr_print_env(fp, vftrace.environment);
   vftr_check_env_names(fp, &vftrace.environment);

   vftr_write_logfile_summary(fp, vftrace.process, runtime);

   vftr_write_logfile_profile_table(fp, vftrace.process.stacktree,
                                    vftrace.environment, runtime);

   vftr_write_logfile_global_stack_list(fp, vftrace.process.collated_stacktree);

   fclose(fp);
   free(logfilename);
}
