/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#define _GNU_SOURCE

#include <stdlib.h>
#include <string.h>
#include <malloc.h>

#include <mpi.h>

#include "vftr_setup.h"
#include "vftr_environment.h"
#include "vftr_timer.h"
#include "vftr_mallinfo.h"

int vftr_xml_string_length;
int vftr_meminfo_method;
vftr_mallinfo_t vftr_current_mallinfo;
bool vftr_memtrace;
int vftr_mmap_xml_index;
long long vftr_mallinfo_ovhd;
long long vftr_mallinfo_post_ovhd;
FILE *vftr_fp_selfstat;

void vftr_init_mallinfo () {
   //printf ("CHECK ENVI: %s\n", vftr_environment.meminfo_method->value);
   vftr_fp_selfstat = NULL;
   if (vftr_environment.meminfo_method->set) {
     memset (&vftr_current_mallinfo, 0, sizeof(vftr_mallinfo_t));
     vftr_mallinfo_ovhd = 0;
     vftr_mallinfo_post_ovhd = 0;
     if (!strcmp (vftr_environment.meminfo_method->value, "MALLOC_INFO")) {
       vftr_meminfo_method = MEM_MALLOC_INFO;
       vftr_memtrace = true;
     } else if (!strcmp (vftr_environment.meminfo_method->value, "SELFSTAT")) {
       vftr_meminfo_method = MEM_SELFSTAT;
       vftr_fp_selfstat = fopen ("/proc/self/statm", "r");
       vftr_memtrace = true;
     } else {
       vftr_memtrace = false;
     }
   }

   // Make a dummy call to malloc_info to determine the number of lines in the XML string and the various indices.
   //vftr_xml_string_length = 0;   
   //char *buf;
   //size_t bufsize;
   //FILE *fp = open_memstream (&buf, &bufsize);
   //malloc_info (0, fp);
   //fclose(fp);
   //
   //vftr_mmap_xml_index = 0;
   //char *token = strtok(buf, "\n");
   //int i = 0;
   //while (token != NULL) {
   //   if (vftr_mpirank == 0) printf ("Init token: %d %s\n", i, token);
   //   if (strstr(token, "type=\"mmap\"") != NULL) vftr_mmap_xml_index = i;
   //   i++;
   //   token = strtok(NULL, "\n");
   //}
   //if (vftr_mmap_xml_index > 0) {
   //  vftr_xml_string_length = i;
   //  vftr_memtrace = true;
   //}
   //if (vftr_mpirank == 0) printf ("---------------------------------------------------\n");
}

void vftr_finalize_mallinfo() {
   if (vftr_fp_selfstat != NULL) fclose (vftr_fp_selfstat);
   vftr_fp_selfstat = NULL;
   vftr_memtrace = false;
}

void vftr_process_mallinfo_line(char *line, long *count, long *size) {
   //if (vftr_mpirank == 0) printf ("THIS LINE: %s\n", line);
   char *tmp = strtok(line, "=\"");
   tmp = strtok(NULL, "\"");
   //printf ("FIRST: %s\n", tmp);
   tmp = strtok(NULL, "=\"");
   //printf ("NEXT 1: %s\n", tmp);
   tmp = strtok(NULL, "\"");
   //printf ("NEXT 2: %s\n", tmp);
   *count = atol(tmp);
   tmp = strtok(NULL, "=\"");
   tmp = strtok(NULL, "\"");
   *size = atol(tmp);
}

void vftr_get_selfstat() {
   //int pid;
   //char *comm;
   //char state;
   //int ppid;
   //int pgrp;
   //int session;
   //int tty_nr;
   //int tpgid;
   //unsigned int flags;
   //unsigned long minflt;
   //unsigned long cminflt;
   //unsigned long majflt;
   //unsigned long cmajflt;
   //unsigned long utime;
   //unsigned long stime;
   //long cutime;
   //long cstime;
   //long priority;
   //long nice;
   //long num_threads;
   //long itrealvalue;
   //unsigned long long starttime;
   //unsigned long vsize;
   //long rss;
   //unsigned long rsslim;
   //unsigned long startcode;
   //unsigned long encode;
   //unsigned long startstack;
   //unsigned long kstkesp;
   //unsigned long kstkeip;
   //unsigned long signal;
   //unsigned long blocked;
   //unsigned long sigignore;
   //unsigned long sigcatch;
   //unsigned long wchan;
   //unsigned long nswap;
   //unsigned long cnswap;
   //int exit_signal;
   //int processor;
   //unsigned int rt_priority;
   //unsigned int policy;
   //unsigned long long delaycct_blkio_ticks;
   //unsigned long guest_time;
   //long cguest_time;
   //unsigned long start_data;
   //unsigned long end_data;
   //unsigned long start_brk;
   //unsigned long arg_start;
   //unsigned long arg_end;
   //unsigned long env_start;
   //unsigned long env_end;
   //int exit_code;
   //FILE *fp = fopen ("/proc/self/stat", "r");
   ////printf ("FILE OPENED\n");
   //fscanf (fp, "%d %s %c %d %d %d %d %d %u %lu %lu %lu %lu %lu %lu %ld %ld %ld %ld %ld %ld %llu %lu %ld %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %lu %d %d %u %u %llu %lu %ld %ld %ld %ld %ld %ld %ld %ld %d",
   //        &pid, &comm, &state, &ppid, &pgrp, &session, &tty_nr, &tpgid, &flags, &minflt, &cminflt, &majflt, &cmajflt,
   //        &utime, &stime, &cutime, &cstime, &priority, &nice, &num_threads, &itrealvalue, &starttime, &vsize,
   //        &rss, &rsslim, &startcode, &encode, &startstack, &kstkesp, &kstkeip, &signal, &blocked, &sigignore, &sigcatch,
   //        &wchan, &nswap, &cnswap, &exit_signal, &processor, &rt_priority, &policy, &delaycct_blkio_ticks,
   //        &guest_time, &cguest_time, &start_data, &end_data, &start_brk, &arg_start, &arg_end, &env_start, &env_end, &exit_code);
   ////printf ("SCANNED FILE!\n");
   //vftr_current_mallinfo.mmap_size = rss;
   //
   int size;
   int resident;
   int shared;
   int text;
   int lib;
   int data;
   int dt;
   //FILE *fp = fopen ("/proc/self/statm", "r");
   fscanf (vftr_fp_selfstat, "%d %d %d %d %d %d %d", &size, &resident, &shared, &text, &lib, &data, &dt);
   //if (vftr_mpirank == 0) {
   //  printf ("%d %d %d %d %d %d %d\n", size, resident, shared, text, lib, data, dt);
   //}
   vftr_current_mallinfo.mmap_size = resident;
   //fclose(fp);
}

void vftr_get_mallinfo () {
   char *buf;
   size_t bufsize;
   vftr_mallinfo_ovhd -= vftr_get_runtime_usec();
   FILE *fp = open_memstream (&buf, &bufsize);  
   malloc_info (0, fp);
   fclose(fp);
   vftr_mallinfo_ovhd += vftr_get_runtime_usec();
 
   vftr_mallinfo_post_ovhd -= vftr_get_runtime_usec();
   char *lines[vftr_xml_string_length];
   char *token = strtok(buf, "\n"); 
   int i = 0;
   while (token != NULL) {
      if (strstr(token, "type=\"mmap\"") != NULL) {
         vftr_process_mallinfo_line (token, &(vftr_current_mallinfo.mmap_count), &(vftr_current_mallinfo.mmap_size));
         break;
      }
      token = strtok(NULL, "\n");
   }
   free(buf);
   vftr_mallinfo_post_ovhd += vftr_get_runtime_usec();
}

void vftr_get_memtrace() {
   switch (vftr_meminfo_method) {
      case MEM_MALLOC_INFO:
         vftr_get_mallinfo();
         break;
      case MEM_SELFSTAT:
         vftr_get_selfstat();
         break;
   }
}


