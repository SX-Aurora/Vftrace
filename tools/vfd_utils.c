#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "vfd_types.h"
#include "sampling_types.h"

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

void print_stack(FILE *fp, unsigned int istack, stack_t *stacklist) {
   fprintf(fp, "%s", stacklist[istack].name);
   if (stacklist[istack].caller >= 0) {
      fprintf(fp, "<");
      print_stack(fp, stacklist[istack].caller, stacklist);
   }
}

void print_stacklist(FILE *fp, unsigned int nstacks, stack_t *stacklist) {
   fprintf(fp, "Stacks list:\n");
   for (unsigned int istack=0; istack<nstacks; istack++) {
      fprintf(fp, "   %u: ", istack);
      print_stack(fp, istack, stacklist);
      fprintf(fp, "\n");
   }
}

void print_function_sample(FILE *vfd_fp, FILE *fp_out,
                           sample_kind kind, stack_t *stacklist) {
   int stackID;
   fread(&stackID, sizeof(int), 1, vfd_fp);
   long long timestamp_usec;
   fread(&timestamp_usec, sizeof(long long), 1, vfd_fp);
   double timestamp = timestamp_usec*1.0e-6;

   fprintf(fp_out, "%16.6f %s ", timestamp,
           kind == samp_function_entry ? "call" : "exit");
   print_stack(fp_out, stackID, stacklist);
   fprintf(fp_out, "\n");
}

void print_samples(FILE *vfd_fp, FILE *fp_out,
                   vfd_header_t vfd_header, stack_t *stacklist) {
   fseek(vfd_fp, vfd_header.samples_offset, SEEK_SET);
   fprintf(fp_out, "Stack and message samples:\n\n");

   unsigned int nsamples = vfd_header.function_samplecount +
                           vfd_header.message_samplecount;
   for (unsigned int isample=0; isample<nsamples; isample++) {
      sample_kind kind;
      fread(&kind, sizeof(sample_kind), 1, vfd_fp);
      switch (kind) {
         case samp_function_entry:
         case samp_function_exit:
            print_function_sample(vfd_fp, fp_out, kind, stacklist);
         break;
         case samp_message:
         break;
         default:
         break;
      }
   }
}
