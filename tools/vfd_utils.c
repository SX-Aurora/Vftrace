#include <stdlib.h>
#include <stdio.h>

#include <string.h>

#include "signal_handling.h"
#include "vfd_types.h"
#include "sampling_types.h"
#include "mpi_util_types.h"

vfd_header_t read_vfd_header(FILE *vfd_fp) {
   vfd_header_t header;
   size_t read_elems;
   read_elems = fread(&(header.vfd_version), sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading vfd_version from vfd-file header\n");
      vftr_abort(0);
   }

   int package_string_len;
   read_elems = fread(&package_string_len, sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading package_string_len from vfd-file header\n");
      vftr_abort(0);
   }
   header.package_string = (char*) malloc(package_string_len*sizeof(char));
   read_elems = fread(header.package_string, sizeof(char), package_string_len, vfd_fp);
   if (read_elems != (size_t)package_string_len) {
      fprintf(stderr, "Error in reading vfd_version from vfd-file header\n");
      vftr_abort(0);
   }

   int datestr_len;
   read_elems = fread(&datestr_len, sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading datestr_len from vfd-file header\n");
      vftr_abort(0);
   }
   header.datestr_start = (char*) malloc(datestr_len*sizeof(char));
   read_elems = fread(header.datestr_start, sizeof(char), datestr_len, vfd_fp);
   if (read_elems != (size_t)datestr_len) {
      fprintf(stderr, "Error in reading datestr_start from vfd-file header\n");
      vftr_abort(0);
   }
   header.datestr_end= (char*) malloc(datestr_len*sizeof(char));
   read_elems = fread(header.datestr_end, sizeof(char), datestr_len, vfd_fp);
   if (read_elems != (size_t)datestr_len) {
      fprintf(stderr, "Error in reading datestr_end from vfd-file header\n");
      vftr_abort(0);
   }

   read_elems = fread(&(header.interval), sizeof(long long), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading interval from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.nprocesses), sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading nprocesses from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.processID), sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading processID from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.nthreads), sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading nthreads from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.runtime), sizeof(double), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading runtime from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.function_samplecount), sizeof(unsigned int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading function_samplecount from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.message_samplecount), sizeof(unsigned int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading message_samplecount from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.nstacks), sizeof(unsigned int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading nstacks from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.n_hw_counters), sizeof(unsigned int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf (stderr, "Error in reader n_hw_counters from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.n_hw_observables), sizeof(unsigned int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf (stderr, "Error in reader n_hw_observables from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.samples_offset), sizeof(long int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading samples_offset from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.stacks_offset), sizeof(long int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading stacks_offset from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.threadtree_offset), sizeof(long int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading threadtree_offset from vfd-file header\n");
      vftr_abort(0);
   }
   read_elems = fread(&(header.hwprof_offset), sizeof(long int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading hwprof_offset from vfd-file header\n");
      vftr_abort(0);
   }

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
   fprintf(vfd_fp, "Processes:       %d of %d\n", vfd_header.processID, vfd_header.nprocesses);
   fprintf(vfd_fp, "Threads:         %d\n", vfd_header.nthreads);
   fprintf(vfd_fp, "Sample interval: %12.6le seconds\n", vfd_header.interval*1.0e-9);
   fprintf(vfd_fp, "Job runtime:     %.3lf seconds\n", vfd_header.runtime);
   fprintf(vfd_fp, "Samples:         %u\n", vfd_header.function_samplecount +
                                            vfd_header.message_samplecount);
   fprintf(vfd_fp, "   Function:     %u\n", vfd_header.function_samplecount );
   fprintf(vfd_fp, "   Messages:     %u\n", vfd_header.message_samplecount );
   fprintf(vfd_fp, "Unique stacks:   %u\n", vfd_header.nstacks);
   fprintf(vfd_fp, "HW counters:     %u\n", vfd_header.n_hw_counters);
   fprintf(vfd_fp, "HW observables:  %u\n", vfd_header.n_hw_observables);
   fprintf(vfd_fp, "Stacks offset:   0x%lx\n", vfd_header.stacks_offset);
   fprintf(vfd_fp, "Sample offset:   0x%lx\n", vfd_header.samples_offset);
   fprintf(vfd_fp, "Thread offset:   0x%lx\n", vfd_header.threadtree_offset);
   fprintf(vfd_fp, "HWprof offset:   0x%lx\n", vfd_header.hwprof_offset);
}

bool is_precise (char *s) {
   return s[strlen(s)-1] == '*';
}

vftr_stack_t *read_stacklist(FILE *vfd_fp, long int stacks_offset,
                        unsigned int nstacks) {
   vftr_stack_t *stacklist = (vftr_stack_t*) malloc(nstacks*sizeof(vftr_stack_t));

   // jump to the stacks
   fseek(vfd_fp, stacks_offset, SEEK_SET);

   // first function is the init with caller id -1
   stacklist[0].ncallees = 0;
   size_t read_elems;
   read_elems = fread(&(stacklist[0].caller), sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading first stacklist caller\n");
      vftr_abort(0);
   }
   int namelen;
   read_elems = fread(&namelen, sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading first stacks namelen\n");
      vftr_abort(0);
   }
   stacklist[0].name = (char*) malloc(namelen*sizeof(char));
   read_elems = fread(stacklist[0].name, sizeof(char), namelen, vfd_fp);
   if (read_elems != (size_t)namelen) {
      fprintf(stderr, "Error in reading first stacks name\n");
      vftr_abort(0);
   }

   // all other stacks
   for (unsigned int istack=1; istack<nstacks; istack++) {
      stacklist[istack].ncallees = 0;
      read_elems = fread(&(stacklist[istack].caller), sizeof(int), 1, vfd_fp);
      if (read_elems != 1) {
         fprintf(stderr, "Error in reading caller of stack %d from vfd-file\n",
                 istack);
         vftr_abort(0);
      }
      // count the number of callees a function has
      stacklist[stacklist[istack].caller].ncallees++;

      // read the functions name
      int namelen;
      read_elems = fread(&namelen, sizeof(int), 1, vfd_fp);
      if (read_elems != 1) {
         fprintf(stderr, "Error in reading namelen of stack %d from vfd-file\n",
                 istack);
         vftr_abort(0);
      }
      stacklist[istack].name = (char*) malloc(namelen*sizeof(char));
      read_elems = fread(stacklist[istack].name, sizeof(char), namelen, vfd_fp);
      if (read_elems != (size_t)namelen) {
         fprintf(stderr, "Error in reading name of stack %d from vfd-file\n",
                 istack);
         vftr_abort(0);
      }

      stacklist[istack].precise = is_precise(stacklist[istack].name);
   }

   // store all the caller callee connections
   // missuse the ncallees value as counter
   // Again the init function needs to be treated separately
   if (stacklist[0].ncallees > 0) {
      stacklist[0].callees = (int*) malloc(stacklist[0].ncallees*sizeof(int));
      stacklist[0].ncallees = 0;
      for (unsigned int istack=1; istack<nstacks; istack++) {
         if (stacklist[istack].ncallees > 0) {
            stacklist[istack].callees =
               (int*) malloc(stacklist[istack].ncallees*sizeof(int));
            stacklist[istack].ncallees = 0;
         }
         // register the current stack as callee of the caller
         vftr_stack_t *caller = stacklist+stacklist[istack].caller;
         caller->callees[caller->ncallees] = istack;
         caller->ncallees++;
      }
   }

   return stacklist;
}

void free_stacklist(unsigned int nstacks, vftr_stack_t *stacklist) {
   for (unsigned int istack=0; istack<nstacks; istack++) {
      free(stacklist[istack].name);
      if (stacklist[istack].ncallees > 0) {
         free(stacklist[istack].callees);
      }
   }
   free(stacklist);
}

void print_stack(FILE *out_fp, unsigned int istack, vftr_stack_t *stacklist) {
   fprintf(out_fp, "%s", stacklist[istack].name);
   if (stacklist[istack].caller >= 0) {
      fprintf(out_fp, "<");
      print_stack(out_fp, stacklist[istack].caller, stacklist);
   }
}

void print_stacklist(FILE *out_fp, unsigned int nstacks, vftr_stack_t *stacklist) {
   fprintf(out_fp, "Stacks list:\n");
   for (unsigned int istack=0; istack<nstacks; istack++) {
      fprintf(out_fp, "   %u: ", istack);
      print_stack(out_fp, istack, stacklist);
      fprintf(out_fp, "\n");
   }
}

thread_t *read_threadtree(FILE *vfd_fp, long int threadtree_offset,
                          int nthreads) {
   thread_t *threadtree = (thread_t*) malloc(nthreads*sizeof(thread_t));

   // jump to the threadtree
   fseek(vfd_fp, threadtree_offset, SEEK_SET);

   // first thread is the root thread with parent id -1
   threadtree[0].nchildren = 0;
   size_t read_elems;
   read_elems = fread(&(threadtree[0].parent_thread), sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading parent_thread of thread %d from vfd-file\n",
              0);
      vftr_abort(0);
   }
   threadtree[0].children = NULL;
   threadtree[0].level = 0;
   threadtree[0].threadID = 0;

   // all other threads
   for (int ithread=1; ithread<nthreads; ithread++) {
      threadtree[ithread].threadID = ithread;
      threadtree[ithread].nchildren = 0;
      int parent_thread = 0;
      read_elems = fread(&parent_thread, sizeof(int), 1, vfd_fp);
      if (read_elems != 1) {
         fprintf(stderr, "Error in reading parent_thread of thread %d from vfd-file\n",
                 ithread);
         vftr_abort(0);
      }
      threadtree[ithread].parent_thread = parent_thread;
      // increment the number of children the parent has
      threadtree[parent_thread].nchildren++;
      // set the level
      threadtree[ithread].level = threadtree[parent_thread].level + 1;
   }

   // store all parent--children relationships
   // missuse the nchildren as counter
   // Again the root thread needs to be treated separately
   if (threadtree[0].nchildren > 0) {
      threadtree[0].children = (int*) malloc(threadtree[0].nchildren*sizeof(int));
      threadtree[0].nchildren = 0;
      for (int ithread=1; ithread<nthreads; ithread++) {
         if (threadtree[ithread].nchildren > 0) {
            threadtree[ithread].children =
               (int*) malloc(threadtree[ithread].nchildren*sizeof(int));
            threadtree[ithread].nchildren = 0;
         }
         // register the current thread as child of the parent
         thread_t *parent = threadtree+threadtree[ithread].parent_thread;
         parent->children[parent->nchildren] = ithread;
         parent->nchildren++;
      }
   }

   return threadtree;
}

void read_hwprof (FILE *vfd_fp, long int hwprof_offset, 
                      int n_counters, int n_observables,
                      char **hwc_names, char **symbols,
                      char **obs_names, char **formulas,
                      char **units) {
   fseek (vfd_fp, hwprof_offset, SEEK_SET);

   size_t read_elems;
   int namelen, symlen;
   for (int i = 0; i < n_counters; i++) { 
      read_elems = fread(&namelen, sizeof(int), 1, vfd_fp);
      hwc_names[i] = (char*)malloc(namelen * sizeof(char));
      read_elems = fread(hwc_names[i], sizeof(char), namelen, vfd_fp);
      read_elems = fread(&symlen, sizeof(int), 1, vfd_fp);
      symbols[i] = (char*)malloc(symlen * sizeof(char));
      read_elems = fread(symbols[i], sizeof(char), symlen, vfd_fp);
   }

   int formlen, unitlen;
   for (int i = 0; i < n_observables; i++) {
      read_elems = fread(&namelen, sizeof(int), 1, vfd_fp);
      obs_names[i] = (char*)malloc(namelen * sizeof(char));
      read_elems = fread(obs_names[i], sizeof(char), namelen, vfd_fp);
      read_elems = fread(&formlen, sizeof(int), 1, vfd_fp);
      formulas[i] = (char*)malloc(formlen * sizeof(char));
      read_elems = fread(formulas[i], sizeof(char), formlen, vfd_fp);
      read_elems = fread(&unitlen, sizeof(int), 1, vfd_fp);
      if (unitlen > 0) {
         units[i] = (char*)malloc(unitlen * sizeof(char));
         read_elems = fread(units[i], sizeof(char), unitlen, vfd_fp);
      } else {
         units[i] = NULL;
      }
   }
}

void free_threadtree(int nthreads, thread_t *threadtree) {
   for (int ithread=0; ithread<nthreads; ithread++) {
      if (threadtree[ithread].nchildren > 0) {
         free(threadtree[ithread].children);
      }
   }
   free(threadtree);
}

void print_thread(FILE *out_fp, int ithread, thread_t *threadtree) {
   thread_t *thread = threadtree+ithread;
   // print level dependent indentation
   for (int ilevel=0; ilevel<thread->level; ilevel++) {
      fprintf(out_fp, "  ");
   }
   fprintf(out_fp, "%d\n", thread->threadID);
   for (int ichild=0; ichild<thread->nchildren; ichild++) {
      print_thread(out_fp, thread->children[ichild], threadtree);
   }
}

void print_threadtree(FILE *out_fp, thread_t *threadtree) {
   fprintf(out_fp, "Threadtree:\n");
   print_thread(out_fp, 0, threadtree);
}

void print_function_sample(FILE *vfd_fp, FILE *out_fp,
                           int n_hwc,
                           sample_kind kind, vftr_stack_t *stacklist) {
   int stackID;
   size_t read_elems;
   read_elems = fread(&stackID, sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading stackID from function_sample from vfd-file\n");
      vftr_abort(0);
   }
   long long timestamp_nsec;
   read_elems = fread(&timestamp_nsec, sizeof(long long), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading timestamp from function_sample from vfd-file\n");
      vftr_abort(0);
   }
   double timestamp_sec = timestamp_nsec*1.0e-9;

   if (n_hwc == 0) {
      fprintf(out_fp, "%16.6f %s ", timestamp_sec,
              kind == samp_function_entry ? "call" : "exit");
   } else {
      long long *counters = (long long*)malloc(n_hwc * sizeof(long long));
      for (int i = 0; i < n_hwc; i++) {
         read_elems = fread(&(counters[i]), sizeof(long long), 1, vfd_fp);
      }
      fprintf (out_fp, "%16.6f ", timestamp_sec);
      for (int i = 0; i < n_hwc; i++) {
         fprintf (out_fp, "%lld ", counters[i]);
      }
      fprintf (out_fp, "%s ", kind == samp_function_entry ? "call" : "exit");
      free(counters);
   }

   print_stack(out_fp, stackID, stacklist);
   fprintf(out_fp, "\n");
}

void print_message_sample(FILE *vfd_fp, FILE *out_fp) {
   message_direction dir;
   size_t read_elems;
   read_elems = fread(&dir, sizeof(message_direction), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading direction from message_sample from vfd-file\n");
      vftr_abort(0);
   }
   int rank;
   read_elems = fread(&rank, sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading rank from message_sample from vfd-file\n");
      vftr_abort(0);
   }
   int type_idx;
   read_elems = fread(&type_idx, sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading type index from message_sample from vfd-file\n");
      vftr_abort(0);
   }
   int count;
   read_elems = fread(&count, sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading count from message_sample from vfd-file\n");
      vftr_abort(0);
   }
   int type_size;
   read_elems = fread(&type_size, sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading type size from message_sample from vfd-file\n");
      vftr_abort(0);
   }
   int tag;
   read_elems = fread(&tag, sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading tag from message_sample from vfd-file\n");
      vftr_abort(0);
   }
   long long tstart, tend;
   read_elems = fread(&tstart, sizeof(long long), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading starting timestamp from message_sample from vfd-file\n");
      vftr_abort(0);
   }
   read_elems = fread(&tend, sizeof(long long), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading ending timestamp from message_sample from vfd-file\n");
      vftr_abort(0);
   }
   int stackID;
   read_elems = fread(&stackID, sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading stackID from message_sample from vfd-file\n");
      vftr_abort(0);
   }
   int threadID;
   read_elems = fread(&threadID, sizeof(int), 1, vfd_fp);
   if (read_elems != 1) {
      fprintf(stderr, "Error in reading threadID from message_sample from vfd-file\n");
      vftr_abort(0);
   }
   double dtstart = tstart*1.0e-9;
   double dtend = tend*1.0e-9;
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
                   vfd_header_t vfd_header, vftr_stack_t *stacklist) {
   fseek(vfd_fp, vfd_header.samples_offset, SEEK_SET);
   fprintf(out_fp, "Stack and message samples:\n");

   unsigned int nsamples = vfd_header.function_samplecount +
                           vfd_header.message_samplecount;
   for (unsigned int isample=0; isample<nsamples; isample++) {
      sample_kind kind;
      size_t read_elems;
      read_elems = fread(&kind, sizeof(sample_kind), 1, vfd_fp);
      if (read_elems != 1) {
         fprintf(stderr, "Error in reading sample kind from vfd-file\n");
         vftr_abort(0);
      }
      switch (kind) {
         case samp_function_entry:
         case samp_function_exit:
            print_function_sample(vfd_fp, out_fp, vfd_header.n_hw_counters, kind, stacklist);
         break;
         case samp_message:
            print_message_sample(vfd_fp, out_fp);
         break;
         default:
         break;
      }
   }
}
