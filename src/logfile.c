#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "self_profile.h"
#include "environment_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "environment.h"
#include "logfile_header.h"
#include "logfile_prof_table.h"
#include "logfile_mpi_table.h"
#include "logfile_stacklist.h"
#include "search.h"
#include "range_expand.h"

char *vftr_get_logfile_name(environment_t environment) {
   char *filename_base = vftr_create_filename_base(environment, -1, 1);
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

FILE *vftr_open_logfile(char *filename) {
   FILE *fp = fopen(filename, "w");
   if (fp == NULL) {
      perror(filename);
      abort();
   }
   return fp;
}

void vftr_write_logfile(vftrace_t vftrace, long long runtime) {
   SELF_PROFILE_START_FUNCTION;
   // only process 0 writes the summary logfile
   if (vftrace.process.processID != 0) {
      SELF_PROFILE_END_FUNCTION;
      return;
   }

   char *logfilename = vftr_get_logfile_name(vftrace.environment);
   FILE *fp = vftr_open_logfile(logfilename);

   vftr_write_logfile_header(fp, vftrace.timestrings);

   vftr_write_logfile_summary(fp, vftrace.process,
                              vftrace.size, runtime);

   vftr_write_logfile_profile_table(fp, vftrace.process.collated_stacktree,
                                    vftrace.environment);

#ifdef _MPI
   vftr_write_logfile_mpi_table(fp, vftrace.process.collated_stacktree,
                                vftrace.environment);
#endif

   vftr_write_logfile_global_stack_list(fp, vftrace.process.collated_stacktree);

   // print environment info
   vftr_print_environment(fp, vftrace.environment);

   fclose(fp);
   free(logfilename);
   SELF_PROFILE_END_FUNCTION;
}
