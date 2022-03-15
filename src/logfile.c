#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <libgen.h>

#include "environment_types.h"
#include "vftrace_state.h"

#include "exe_info.h"
#include "misc_utils.h"
#include "timer_types.h"
#include "license.h"
#include "config.h"

char *vftr_get_logfile_name(environment_t environment, int rankID, int nranks) {
   if (environment.logfile_basename.set) {
      // user defined logfile name
      return strdup(environment.logfile_basename.value.string_val);
   } else {
      // default name constructed from executable name
      char *exe_path = vftr_get_exectuable_path();
      if (exe_path == NULL) {
         return strdup("unknown");
      } else {
         // name will be <exe_name>_<rankID>.log
         // exe_name:
         char *exe_name = basename(exe_path);
         int exe_name_len = strlen(exe_name);
         // rankID (leading zeroes for nice sorting among ranks):
         int ndigits = vftr_count_base_digits(nranks, 10);
         int rankID_len = snprintf(NULL, 0, "%0*d", ndigits, rankID);
         // extension .log
         char *extension = ".log";
         int extension_len = strlen(extension);

         // construct pathname
         int total_len = exe_name_len +
                         1 + // underscore
                         rankID_len +
                         extension_len +
                         1; // null terminator
         char *logfile_name = (char*) malloc(total_len*sizeof(char));
         strcpy(logfile_name, exe_name);
         logfile_name[exe_name_len] = '_';
         snprintf(logfile_name+exe_name_len+1, rankID_len+1, "%0*d", ndigits, rankID);
         strcat(logfile_name, extension);

         free(exe_path);
         return logfile_name;
      }
   }
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

void vftr_write_logfile(vftrace_t vftrace) {
   char *logfilename = vftr_get_logfile_name(vftrace.environment,
                                             vftrace.process.processID,
                                             vftrace.process.nprocesses);
   FILE *fp = fopen(logfilename, "w");
   if (fp == NULL) {
      fprintf(stderr, "Unable to open \"%s\" for writing.\n", logfilename);
      return;
   }

   vftr_write_logfile_header(fp, vftrace.timestrings,
                             vftrace.environment);





   fclose(fp);
   free(logfilename);
}
