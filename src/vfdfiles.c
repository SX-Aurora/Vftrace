#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <unistd.h>
#include <assert.h>

#include "environment_types.h"

#include "filenames.h"

// At the initialization of vftrace the mpi-rank and comm-size is
// not known for paralle programs.
// Thus a preliminary vfdfile is created:
// <basename>_<pid>.tmpvfd
// In the finalization it will be moved to its proper name
// <basename>_<mpi-rank>.vfd
char *vftr_get_preliminary_vfdfile_name(environment_t environment) {
   int pid = getpid();
   char *filename_base = vftr_create_filename_base(environment, pid, pid);
   int filename_base_len = strlen(filename_base);

   char *extension = ".tmpvfd";
   int extension_len = strlen(extension);

   // construct preliminary vfdfile name
   int total_len = filename_base_len +
                   extension_len +
                   1; // null terminator
   char *vfdfile_name = (char*) malloc(total_len*sizeof(char));
   strcpy(vfdfile_name, filename_base);
   strcat(vfdfile_name, extension);

   free(filename_base);
   return vfdfile_name;
}

char *vftr_get_vfdfile_name(environment_t environment, int rankID, int nranks) {
   char *filename_base = vftr_create_filename_base(environment, rankID, nranks);
   int filename_base_len = strlen(filename_base);

   char *extension = ".vfd";
   int extension_len = strlen(extension);
   
   // construct preliminary vfdfile name
   int total_len = filename_base_len +
                   extension_len +
                   1; // null terminator
   char *vfdfile_name = (char*) malloc(total_len*sizeof(char));
   strcpy(vfdfile_name, filename_base);
   strcat(vfdfile_name, extension);
   
   free(filename_base);
   return vfdfile_name;
}

FILE *vftr_open_vfdfile(char *filename) {
   FILE *fp = fopen(filename, "w+");
   if (fp == NULL) {
      perror(filename);
      abort();
   }
   return fp;
}

char *vftr_attach_iobuffer_vfdfile(FILE *fp, environment_t environment) {
   // the environment variable gets the size in MB.
   size_t bufsize = environment.bufsize.value.int_val;
   // it is needed in bytes
   bufsize *= 1024*1024;
   char *buffer = (char*) malloc(bufsize);
   assert(buffer);
   int status = setvbuf(fp, buffer, _IOFBF, bufsize);
   assert(!status);
   return buffer;
}

int vftr_rename_vfdfile(char *prelim_name, char *final_name) {
   int error = rename(prelim_name, final_name);
   if (error != 0) {
      perror(final_name);
   }
   return error;
}
