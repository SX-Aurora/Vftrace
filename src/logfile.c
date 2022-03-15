#include <stdlib.h>
#include <stdio.h>

#include <unistd.h>
#include <string.h>
#include <libgen.h>

#include "vftrace_state.h"
#include "misc_utils.h"

char *vftr_get_exectuable_path() {
   // get the file that contains
   // the commandline for this process
   int pid = getpid();
   char *proccmdfmt = "/proc/%d/cmdline";
   int proccmd_len = snprintf(NULL, 0, proccmdfmt, pid) + 1;
   char *proccmd = (char*) malloc(proccmd_len*sizeof(char));
   snprintf(proccmd, proccmd_len, proccmdfmt, pid);

   // read the commandline from file
   FILE *cmdlinefp = fopen(proccmd, "r");
   free(proccmd);
   if (cmdlinefp == NULL) {
      return NULL;
   } else {
      char *cmdline = NULL;
      size_t cmdline_len = 0;
      getline(&cmdline,&cmdline_len,cmdlinefp);
#ifdef __ve__
      // get position where the actual command starts
      // skipping veexec and its options
      char *cmdlineend = cmdline+cmdline_len;
      int found = 0;
      while (cmdline < cmdlineend && !found) {
         found = !strcmp(cmdline, "--");
         while (*cmdline != '\0') {
            cmdline++;
         }
         cmdline++;
      }
#endif
      fclose(cmdlinefp);
      return cmdline;
   }
}

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

void vftr_write_logfile(vftrace_t vftrace) {
   char *logfilename = vftr_get_logfile_name(vftrace.environment,
                                             vftrace.process.processID,
                                             vftrace.process.nprocesses);
   FILE *fp = fopen(logfilename, "w");
   if (fp == NULL) {
      fprintf(stderr, "Unable to open \"%s\" for writing.\n", logfilename);
   }



   printf("logfilename: %s\n", logfilename);

   fclose(fp);
   free(logfilename);
}
