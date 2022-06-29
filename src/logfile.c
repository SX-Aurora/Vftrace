#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "environment_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "environment.h"
#include "logfile_header.h"
#include "logfile_prof_table.h"
#include "logfile_stacklist.h"

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
                                    vftrace.environment);



   vftr_write_logfile_global_stack_list(fp, vftrace.process.collated_stacktree);

   fclose(fp);
   free(logfilename);
}
