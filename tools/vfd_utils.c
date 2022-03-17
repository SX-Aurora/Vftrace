#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "vfd_types.h"


#ifndef VFD_TYPES_H
#define VFD_TYPES_H

#include <stdbool.h>

typedef struct {
   int vfd_version;
   char *package_string;
   char *datestr_start;
   char *datestr_end;
   long long interval;
   unsigned int nprocesses;
   unsigned int processID;
   double runtime;
   unsigned int function_samplecount;
   unsigned int message_samplecount;
   unsigned int nstacks;
   long int samples_offset;
   long int stacks_offset;
   // TODO: hardware counters
} vfd_header_t;

typedef struct {
    char *name;
    int caller;
    int ncallees;
    int *callees;
    bool precise;
} stack_entry_t;

#endif
vfd_header_t read_vfd_header(FILE *fp) {
   vfd_header_t header;
   fread(&(header.vfd_version), sizeof(int), 1, fp);

   int package_string_len;
   fread(&package_string_len, sizeof(int), 1, fp);
   header.package_string = (char*) malloc(package_string_len*sizeof(char));
   fread(header.package_string, sizeof(char), package_string_len, fp);

   int datestr_len;
   fread(&datestr_len, sizeof(int), 1, fp);
   header.datestr_start = (char*) malloc(datestr_len*sizeof(char));
   fread(header.datestr_start, sizeof(char), datestr_len, fp);
   header.datestr_end= (char*) malloc(datestr_len*sizeof(char));
   fread(header.datestr_end, sizeof(char), datestr_len, fp);

   fread(&(header.interval), sizeof(long long), 1, fp);
   fread(&(header.nprocesses), sizeof(unsigned int), 1, fp);
   fread(&(header.processID), sizeof(unsigned int), 1, fp);
   fread(&(header.runtime), sizeof(double), 1, fp);
   fread(&(header.function_samplecount), sizeof(unsigned int), 1, fp);
   fread(&(header.message_samplecount), sizeof(unsigned int), 1, fp);
   fread(&(header.nstacks), sizeof(unsigned int), 1, fp);
   fread(&(header.samples_offset), sizeof(long int), 1, fp);
   fread(&(header.stacks_offset), sizeof(long int), 1, fp);

   return header;
}

void free_vfd_header(vfd_header_t *vfd_header) {
   free(vfd_header->package_string);
   free(vfd_header->datestr_start);
   free(vfd_header->datestr_end);
}

void print_vfd_header(FILE *fp, vfd_header_t vfd_header) {
   fprintf(fp, "Version ID:      %s\n", vfd_header.package_string);
   fprintf(fp, "Start Date:      %s\n", vfd_header.datestr_start);
   fprintf(fp, "End Date:        %s\n", vfd_header.datestr_end);
   fprintf(fp, "Processes:       %u of %u\n", vfd_header.processID, vfd_header.nprocesses);
   fprintf(fp, "Sample interval: %12.6le seconds\n", vfd_header.interval*1.0e-6);
   fprintf(fp, "Job runtime:     %.3lf seconds\n", vfd_header.runtime);
   fprintf(fp, "Samples:         %u\n", vfd_header.function_samplecount + 
                                        vfd_header.message_samplecount);
   fprintf(fp, "   Function:     %u\n", vfd_header.function_samplecount );
   fprintf(fp, "   Messages:     %u\n", vfd_header.message_samplecount );
   fprintf(fp, "Unique stacks:   %u\n", vfd_header.nstacks);
   fprintf(fp, "Stacks offset:   0x%lx\n", vfd_header.stacks_offset);
   fprintf(fp, "Sample offset:   0x%lx\n", vfd_header.samples_offset);
}

bool is_precise (char *s) {
   return s[strlen(s)-1] == '*';
}

stack_t *read_stacklist(FILE *fp, long int stacks_offset,
                        unsigned int nstacks) {
   stack_t *stacklist = (stack_t*) malloc(nstacks*sizeof(stack_t));

   // jump to the stacks
   fseek(fp, stacks_offset, SEEK_SET);

   // first function is the init with caller id -1
   stacklist[0].ncallees = 0;
   fread(&(stacklist[0].caller), sizeof(int), 1, fp);
   int namelen;
   fread(&namelen, sizeof(int), 1, fp);
   stacklist[0].name = (char*) malloc(namelen*sizeof(char));
   fread(stacklist[0].name, sizeof(char), namelen, fp);

   // all other stacks
   for (unsigned int istack=1; istack<nstacks; istack++) {
      stacklist[istack].ncallees = 0;
      fread(&(stacklist[istack].caller), sizeof(int), 1, fp);
      // count the number of callees a function has
      stacklist[stacklist[istack].caller].ncallees++;

      // read the functions name
      int namelen;
      fread(&namelen, sizeof(int), 1, fp);
      stacklist[istack].name = (char*) malloc(namelen*sizeof(char));
      fread(stacklist[istack].name, sizeof(char), namelen, fp);

      stacklist[istack].precise = is_precise(stacklist[istack].name);
   }

   // store all the caller callee connections
   // missuse the ncallees value as counter
   // Again the init function needs to be treated separately
   stacklist[0].callees = (int*) malloc(stacklist[0].ncallees*sizeof(int));
   stacklist[0].ncallees = 0;
   for (unsigned int istack=1; istack<nstacks; istack++) {
      stacklist[istack].callees = (int*) malloc(stacklist[istack].ncallees*sizeof(int));
      stacklist[istack].ncallees = 0;
      // register the current stack as callee of the caller
      stack_t *caller = stacklist+stacklist[istack].caller;
      caller->callees[caller->ncallees] = istack;
      caller->ncallees++;
   }

   return stacklist;
}

void free_stacklist(unsigned int nstacks, stack_t *stacklist) {
   for (unsigned int istack=0; istack<nstacks; istack++) {
      free(stacklist[istack].name);
      free(stacklist[istack].callees);
   }
   free(stacklist);
}
