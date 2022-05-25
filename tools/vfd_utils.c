#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "vfd_types.h"
#include "sampling_types.h"
#include "mpi_util_types.h"

vfd_header_t read_vfd_header(FILE *vfd_fp) {
   vfd_header_t header;
   fread(&(header.vfd_version), sizeof(int), 1, vfd_fp);

   int package_string_len;
   fread(&package_string_len, sizeof(int), 1, vfd_fp);
   header.package_string = (char*) malloc(package_string_len*sizeof(char));
   fread(header.package_string, sizeof(char), package_string_len, vfd_fp);

   int datestr_len;
   fread(&datestr_len, sizeof(int), 1, vfd_fp);
   header.datestr_start = (char*) malloc(datestr_len*sizeof(char));
   fread(header.datestr_start, sizeof(char), datestr_len, vfd_fp);
   header.datestr_end= (char*) malloc(datestr_len*sizeof(char));
   fread(header.datestr_end, sizeof(char), datestr_len, vfd_fp);

   fread(&(header.interval), sizeof(long long), 1, vfd_fp);
   fread(&(header.nprocesses), sizeof(unsigned int), 1, vfd_fp);
   fread(&(header.processID), sizeof(unsigned int), 1, vfd_fp);
   fread(&(header.runtime), sizeof(double), 1, vfd_fp);
   fread(&(header.function_samplecount), sizeof(unsigned int), 1, vfd_fp);
   fread(&(header.message_samplecount), sizeof(unsigned int), 1, vfd_fp);
   fread(&(header.nstacks), sizeof(unsigned int), 1, vfd_fp);
   fread(&(header.samples_offset), sizeof(long int), 1, vfd_fp);
   fread(&(header.stacks_offset), sizeof(long int), 1, vfd_fp);

   return header;
}

void free_vfd_header(vfd_header_t *vfd_header) {
   free(vfd_header->package_string);
   free(vfd_header->datestr_start);
   free(vfd_header->datestr_end);
}

void print_vfd_header(FILE *vfd_fp, vfd_header_t vfd_header) {
   fprintf(vfd_fp, "Version ID:      %s\n", vfd_header.package_string);
   fprintf(vfd_fp, "Start Date:      %s\n", vfd_header.datestr_start);
   fprintf(vfd_fp, "End Date:        %s\n", vfd_header.datestr_end);
   fprintf(vfd_fp, "Processes:       %u of %u\n", vfd_header.processID, vfd_header.nprocesses);
   fprintf(vfd_fp, "Sample interval: %12.6le seconds\n", vfd_header.interval*1.0e-6);
   fprintf(vfd_fp, "Job runtime:     %.3lf seconds\n", vfd_header.runtime);
   fprintf(vfd_fp, "Samples:         %u\n", vfd_header.function_samplecount +
                                            vfd_header.message_samplecount);
   fprintf(vfd_fp, "   Function:     %u\n", vfd_header.function_samplecount );
   fprintf(vfd_fp, "   Messages:     %u\n", vfd_header.message_samplecount );
   fprintf(vfd_fp, "Unique stacks:   %u\n", vfd_header.nstacks);
   fprintf(vfd_fp, "Stacks offset:   0x%lx\n", vfd_header.stacks_offset);
   fprintf(vfd_fp, "Sample offset:   0x%lx\n", vfd_header.samples_offset);
}

bool is_precise (char *s) {
   return s[strlen(s)-1] == '*';
}

stack_t *read_stacklist(FILE *vfd_fp, long int stacks_offset,
                        unsigned int nstacks) {
   stack_t *stacklist = (stack_t*) malloc(nstacks*sizeof(stack_t));

   // jump to the stacks
   fseek(vfd_fp, stacks_offset, SEEK_SET);

   // first function is the init with caller id -1
   stacklist[0].ncallees = 0;
   fread(&(stacklist[0].caller), sizeof(int), 1, vfd_fp);
   int namelen;
   fread(&namelen, sizeof(int), 1, vfd_fp);
   stacklist[0].name = (char*) malloc(namelen*sizeof(char));
   fread(stacklist[0].name, sizeof(char), namelen, vfd_fp);

   // all other stacks
   for (unsigned int istack=1; istack<nstacks; istack++) {
      stacklist[istack].ncallees = 0;
      fread(&(stacklist[istack].caller), sizeof(int), 1, vfd_fp);
      // count the number of callees a function has
      stacklist[stacklist[istack].caller].ncallees++;

      // read the functions name
      int namelen;
      fread(&namelen, sizeof(int), 1, vfd_fp);
      stacklist[istack].name = (char*) malloc(namelen*sizeof(char));
      fread(stacklist[istack].name, sizeof(char), namelen, vfd_fp);

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

void print_stack(FILE *out_fp, unsigned int istack, stack_t *stacklist) {
   fprintf(out_fp, "%s", stacklist[istack].name);
   if (stacklist[istack].caller >= 0) {
      fprintf(out_fp, "<");
      print_stack(out_fp, stacklist[istack].caller, stacklist);
   }
}

void print_stacklist(FILE *out_fp, unsigned int nstacks, stack_t *stacklist) {
   fprintf(out_fp, "Stacks list:\n");
   for (unsigned int istack=0; istack<nstacks; istack++) {
      fprintf(out_fp, "   %u: ", istack);
      print_stack(out_fp, istack, stacklist);
      fprintf(out_fp, "\n");
   }
}

void print_function_sample(FILE *vfd_fp, FILE *out_fp,
                           sample_kind kind, stack_t *stacklist) {
   int stackID;
   fread(&stackID, sizeof(int), 1, vfd_fp);
   long long timestamp_usec;
   fread(&timestamp_usec, sizeof(long long), 1, vfd_fp);
   double timestamp = timestamp_usec*1.0e-6;

   fprintf(out_fp, "%16.6f %s ", timestamp,
           kind == samp_function_entry ? "call" : "exit");
   print_stack(out_fp, stackID, stacklist);
   fprintf(out_fp, "\n");
}

void print_message_sample(FILE *vfd_fp, FILE *out_fp) {
   message_direction dir;
   fread(&dir, sizeof(message_direction), 1, vfd_fp);
   int rank;
   fread(&rank, sizeof(int), 1, vfd_fp);
   int type_idx;
   fread(&type_idx, sizeof(int), 1, vfd_fp);
   int count;
   fread(&count, sizeof(int), 1, vfd_fp);
   int type_size;
   fread(&type_size, sizeof(int), 1, vfd_fp);
   int tag;
   fread(&tag, sizeof(int), 1, vfd_fp);
   long long tstart, tend;
   fread(&tstart, sizeof(long long), 1, vfd_fp);
   fread(&tend, sizeof(long long), 1, vfd_fp);
   int stackID;
   fread(&stackID, sizeof(int), 1, vfd_fp);
   int threadID;
   fread(&threadID, sizeof(int), 1, vfd_fp);
   double dtstart = tstart*1.0e-6;
   double dtend = tend*1.0e-6;
   double rate = (count * type_size) / ((dtend - dtstart)*1024.0*1024.0);

   fprintf(out_fp, "%16.6f %s in stackID %d from threadID %d\n",
           dtstart, dir == send ? "send" : "recv", stackID, threadID);
   fprintf(out_fp, "%16s count=%d type=%s(%iBytes) ",
           "", count, vftr_get_mpitype_string_from_idx(type_idx), type_size);
   fprintf(out_fp, "rate= %8.4lf MiB/s peer=%d tag=%d\n",
           rate, rank, tag);
   fprintf(out_fp, "%16.6f %s end\n",
           dtend, dir == send ? "send" : "recv");
}

void print_samples(FILE *vfd_fp, FILE *out_fp,
                   vfd_header_t vfd_header, stack_t *stacklist) {
   fseek(vfd_fp, vfd_header.samples_offset, SEEK_SET);
   fprintf(out_fp, "Stack and message samples:\n");

   unsigned int nsamples = vfd_header.function_samplecount +
                           vfd_header.message_samplecount;
   for (unsigned int isample=0; isample<nsamples; isample++) {
      sample_kind kind;
      fread(&kind, sizeof(sample_kind), 1, vfd_fp);
      switch (kind) {
         case samp_function_entry:
         case samp_function_exit:
            print_function_sample(vfd_fp, out_fp, kind, stacklist);
         break;
         case samp_message:
            print_message_sample(vfd_fp, out_fp);
         break;
         default:
         break;
      }
   }
}
