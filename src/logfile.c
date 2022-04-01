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
#include "tables.h"

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

void vftr_write_logfile_summary(FILE *fp, vftrace_t vftrace, long long runtime) {
   double runtime_sec = runtime * 1.0e-6;
   double overhead_sec = vftr_total_overhead_usec(vftrace.process.stacktree)*1.0e-6;
   double apptime_sec = runtime_sec - overhead_sec;
   fprintf(fp, "------------------------------------------------------------\n");
#ifdef _MPI
   fprintf(fp, "Nr. of MPI ranks      %8d\n",
           vftrace.process.nprocesses);
#endif
   fprintf(fp, "Total runtime:        %8.2f seconds\n", runtime_sec);
   fprintf(fp, "Application time:     %8.2f seconds\n", apptime_sec);
   fprintf(fp, "Overhead:             %8.2f seconds (%.2f%%)\n",
           overhead_sec, 100.0 * overhead_sec / runtime_sec);
   // TODO: Add Sampling overhead
   // TODO: Add MPI overhead
   // TODO: distinguish between overhead from threads and regular overhead
   // TODO: Add Performance counters
   fprintf(fp, "------------------------------------------------------------\n");
}

void vftr_write_logfile_profile_table(FILE *fp, stacktree_t stacktree,
                                      environment_t environment,
                                      long long runtime) {
   fprintf(fp, "\nRuntime profile\n");

   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, stacktree.nstacks);

   int *calls = vftr_stack_calls_list(stacktree.nstacks, stacktree.stacks);
   vftr_table_add_column(&table, col_int, "Calls", "%d", 'c', 'r', (void*) calls);

   double *excl_time = vftr_stack_exclusive_time_list(stacktree.nstacks, stacktree.stacks);
   vftr_table_add_column(&table, col_double, "t_excl/s", "%.3f", 'c', 'r', (void*) excl_time);

   double *incl_time = vftr_stack_inclusive_time_list(stacktree.nstacks, stacktree.stacks);
   vftr_table_add_column(&table, col_double, "t_incl/s", "%.3f", 'c', 'r', (void*) incl_time);
   
  // 
  // double *vftr_stack_overhead_time_list(int nstacks, stack_t *stacks);

   char **function_names = vftr_stack_function_name_list(stacktree.nstacks, stacktree.stacks);
   vftr_table_add_column(&table, col_string, "Function", "%s", 'c', 'r', (void*) function_names);

   char **caller_names = vftr_stack_caller_name_list(stacktree.nstacks, stacktree.stacks);
   vftr_table_add_column(&table, col_string, "Caller", "%s", 'c', 'r', (void*) caller_names);

   vftr_print_table(fp, table);

   vftr_table_free(&table);
   free(calls);
   free(excl_time);
   free(incl_time);
   free(function_names);
   free(caller_names);
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


   vftr_write_logfile_summary(fp, vftrace, runtime);

   vftr_write_logfile_profile_table(fp, vftrace.process.stacktree,
                                    vftrace.environment, runtime);



   fclose(fp);
   free(logfilename);
}
