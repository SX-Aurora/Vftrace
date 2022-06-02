#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <unistd.h>

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
   if (cmdlinefp == NULL) {
      perror(proccmd);
      free(proccmd);
      return NULL;
   } else {
      free(proccmd);
      char *cmdline = NULL;
      size_t cmdline_len = 0;
      ssize_t read_bytes = getline(&cmdline,&cmdline_len,cmdlinefp);
      if (read_bytes < 0) {
         perror("Error");
         return NULL;
      }
      // store memory location for later dealloc
      char *tmpcmdline = cmdline;
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
      cmdline = strdup(cmdline);
      free(tmpcmdline);
      fclose(cmdlinefp);
      return cmdline;
   }
}
