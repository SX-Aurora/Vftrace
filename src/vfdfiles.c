#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <unistd.h>
#include <assert.h>

#include "self_profile.h"
#include "configuration_types.h"
#include "process_types.h"
#include "stack_types.h"
#include "timer_types.h"
#include "vftrace_state.h"

#include "filenames.h"
#include "vfdfiles.h"
#include "timer.h"
#include "config.h"

// At the initialization of vftrace the mpi-rank and comm-size is
// not known for paralle programs.
// Thus a preliminary vfdfile is created:
// <basename>_<pid>.tmpvfd
// In the finalization it will be moved to its proper name
// <basename>_<mpi-rank>.vfd
char *vftr_get_preliminary_vfdfile_name(config_t config) {
   SELF_PROFILE_START_FUNCTION;
   int pid = getpid();
   char *filename_base = vftr_create_filename_base(config, pid, pid);
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
   SELF_PROFILE_END_FUNCTION;
   return vfdfile_name;
}

char *vftr_get_vfdfile_name(config_t config, int rankID, int nranks) {
   SELF_PROFILE_START_FUNCTION;
   char *filename_base = vftr_create_filename_base(config, rankID, nranks);
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
   SELF_PROFILE_END_FUNCTION;
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

char *vftr_attach_iobuffer_vfdfile(FILE *fp, config_t config,
                                   size_t *buffersize) {
   // the configuration variable gets the size in MB.
   size_t bufsize = config.sampling.outbuffer_size.value;
   // it is needed in bytes
   bufsize *= 1024*1024;
   *buffersize = bufsize;
   char *buffer = (char*) malloc(bufsize);
   assert(buffer);
   memset((void*) buffer, 0, bufsize);
   int status = setvbuf(fp, buffer, _IOFBF, bufsize);
   assert(!status);
   return buffer;
}

int vftr_rename_vfdfile(char *prelim_name, char *final_name) {
   SELF_PROFILE_START_FUNCTION;
   int error = rename(prelim_name, final_name);
   if (error != 0) {
      perror(final_name);
   }
   SELF_PROFILE_END_FUNCTION;
   return error;
}

void vftr_write_incomplete_vfd_header(sampling_t *sampling) {
   SELF_PROFILE_START_FUNCTION;
   FILE *fp = sampling->vfdfilefp;

   int zeroint = 0;
   unsigned int zerouint = 0;
   double zerodouble = 0.0;
   long long zerolonglong = 0;

   int vfd_version = VFD_VERSION;
   fwrite(&vfd_version, sizeof(int), 1, fp);

   char *package_string = PACKAGE_STRING;
   int package_string_len = strlen(package_string)+1; // +1 for 0 terminator
   fwrite(&package_string_len, sizeof(int), 1, fp);
   fwrite(package_string, sizeof(char), package_string_len, fp);

   // the datestrings will be written at the end. just reserver the space
   // by creating a dummy datestring.
   char *datestr = vftr_get_date_str();
   int datestr_len = strlen(datestr)+1;
   fwrite(&datestr_len, sizeof(int), 1, fp);
   // Write twice to reserve space for beginning time and end time string
   fwrite(datestr, sizeof(char), datestr_len, fp);
   fwrite(datestr, sizeof(char), datestr_len, fp);
   free(datestr);

   // reserve some space to be filled in later
   // sampling interval in usec
   fwrite(&zerolonglong, sizeof(long long), 1, fp);
   // number of processes
   fwrite(&zeroint, sizeof(int), 1, fp);
   // my process id
   fwrite(&zeroint, sizeof(int), 1, fp);
   // number of threads
   fwrite(&zeroint, sizeof(int), 1, fp);
   // runtime in seconds
   fwrite(&zerodouble, sizeof(double), 1, fp);
   // sample count
   fwrite(&zerouint, sizeof(unsigned int), 1, fp);
   // message sample count
   fwrite(&zerouint, sizeof(unsigned int), 1, fp);
   // stacks count
   fwrite(&zerouint, sizeof(unsigned int), 1, fp);
   // samples offset
   fwrite(&zerolonglong, sizeof(long int), 1, fp);
   // stacks offset
   fwrite(&zerolonglong, sizeof(long int), 1, fp);
   // threadtree offset
   fwrite(&zerolonglong, sizeof(long int), 1, fp);
   // TODO: Add hardware scenarios

   // Now the samples will come,
   // so the current position is the sample offset
   sampling->samples_offset = ftell(fp);
   SELF_PROFILE_END_FUNCTION;
}

void vftr_update_vfd_header(sampling_t *sampling,
                            process_t process,
                            time_strings_t timestrings,
                            double runtime) {
   SELF_PROFILE_START_FUNCTION;
   FILE *fp = sampling->vfdfilefp;
   // jump to the beginning of the file
   fseek(fp, 0, SEEK_SET);

   // skip vfd-version
   fseek(fp, sizeof(int), SEEK_CUR);

   // skip package string
   int package_string_len;
   size_t read_elems = 0;
   read_elems = fread(&package_string_len, sizeof(int), 1, fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading package string len of vfd-file\n");
   }
   fseek(fp, package_string_len*sizeof(char), SEEK_CUR);

   // write the date strings
   int datestr_len;
   read_elems = fread(&datestr_len, sizeof(int), 1, fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading datastr_len of vfd-file\n");
   }
   // check if the datestring length is consistent (+1 for null terminator)
   if ((size_t) (datestr_len) != strlen(timestrings.start_time)+1) {
      fprintf(stderr, "Inconsistency in time string length!\n"
              "VFD-start time string is not updated!\n");
   } else {
      fwrite(timestrings.start_time, sizeof(char), datestr_len, fp);
   }
   if ((size_t) (datestr_len) != strlen(timestrings.end_time)+1) {
      fprintf(stderr, "Inconsistency in time string length!\n"
              "VFD-end time string is not updated!\n");
   } else {
      fwrite(timestrings.end_time, sizeof(char), datestr_len, fp);
   }

   // sampling interval in usec
   fwrite(&(sampling->interval), sizeof(long long), 1, fp);
   // number of processes
   fwrite(&(process.nprocesses), sizeof(int), 1, fp);
   // my process id
   fwrite(&(process.processID), sizeof(int), 1, fp);
   // number of threads
   fwrite(&(process.threadtree.nthreads), sizeof(int), 1, fp);
   // runtime in seconds
   fwrite(&runtime, sizeof(double), 1, fp);
   // sample count
   fwrite(&(sampling->function_samplecount), sizeof(unsigned int), 1, fp);
   // message sample count
   fwrite(&(sampling->message_samplecount), sizeof(unsigned int), 1, fp);
   // stacks count
   fwrite(&(process.stacktree.nstacks), sizeof(unsigned int), 1, fp);
   // samples offset
   fwrite(&(sampling->samples_offset), sizeof(long int), 1, fp);
   // stacks offset
   fwrite(&(sampling->stacktable_offset), sizeof(long int), 1, fp);
   // threadtree offset
   fwrite(&(sampling->threadtree_offset), sizeof(long int), 1, fp);
   SELF_PROFILE_END_FUNCTION;
}

void vftr_write_vfd_stacks(sampling_t *sampling, stacktree_t stacktree) {
   SELF_PROFILE_START_FUNCTION;
   FILE *fp = sampling->vfdfilefp;

   // save the offset of where the stacktable begins
   sampling->stacktable_offset = ftell(fp);

   // to reconstruct the stacktree lateron
   // it is sufficient to know:
   // 1. index of the calling stack
   // 2. the name of the function
   for (int istack=0; istack<stacktree.nstacks; istack++) {
      vftr_stack_t stack = stacktree.stacks[istack];
      fwrite(&(stack.caller), sizeof(int), 1, fp);
      // precisely sampled functions are marked
      // with a '*' after their name
      if (stack.precise) {
         int namelen = strlen(stack.cleanname) + 2; // +1 for '*' and +1 for null terminator
         fwrite(&namelen, sizeof(int), 1, fp);
         fwrite(stack.cleanname, sizeof(char), namelen-2, fp);
         fwrite("*", sizeof(char), 2, fp);
      } else {
         int namelen = strlen(stack.cleanname) + 1; // +1 for null terminator
         fwrite(&namelen, sizeof(int), 1, fp);
         fwrite(stack.cleanname, sizeof(char), namelen, fp);
      }
   }
   SELF_PROFILE_END_FUNCTION;
}

void vftr_write_vfd_threadtree(sampling_t *sampling, threadtree_t threadtree) {
   SELF_PROFILE_START_FUNCTION;
   FILE *fp = sampling->vfdfilefp;

   // save the offset of where the threadtree begins
   sampling->threadtree_offset = ftell(fp);

   // to reconstruct the stacktree later on
   // it is sufficien to know the index
   // of the parent thread
   for (int ithread=0; ithread<threadtree.nthreads; ithread++) {
      thread_t thread = threadtree.threads[ithread];
      fwrite(&(thread.parent_thread), sizeof(int), 1, fp);
   }
   SELF_PROFILE_END_FUNCTION;
}

void vftr_write_vfd_function_sample(sampling_t *sampling, sample_kind kind,
                                    int stackID, long long timestamp) {
   SELF_PROFILE_START_FUNCTION;
   FILE *fp = sampling->vfdfilefp;
   fwrite(&kind, sizeof(sample_kind), 1, fp);
   fwrite(&stackID, sizeof(int), 1, fp);
   fwrite(&timestamp, sizeof(long long), 1, fp);
   SELF_PROFILE_END_FUNCTION;
}
