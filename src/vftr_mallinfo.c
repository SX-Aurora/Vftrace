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

#ifdef _MPI
#include <mpi.h>
#endif

#include "vftr_setup.h"
#include "vftr_environment.h"
#include "vftr_timer.h"
#include "vftr_mallinfo.h"
#include "vftr_functions.h"

int vftr_xml_string_length;
int vftr_meminfo_method;
vftr_mallinfo_t vftr_current_mallinfo;
bool vftr_memtrace;
int vftr_mmap_xml_index;
long long vftr_mallinfo_ovhd;
long long vftr_mallinfo_post_ovhd;
FILE *vftr_fp_vmrss;

/**********************************************************************/

void vftr_init_mallinfo () {
   vftr_fp_vmrss = NULL;
   if (vftr_environment.meminfo_method->set) {
     vftr_mallinfo_ovhd = 0;
     vftr_mallinfo_post_ovhd = 0;
     if (!strcmp (vftr_environment.meminfo_method->value, "MALLOC_INFO")) {
       vftr_meminfo_method = MEM_MALLOC_INFO;
       vftr_memtrace = true;
     } else if (!strcmp (vftr_environment.meminfo_method->value, "VMRSS")) {
       vftr_meminfo_method = MEM_VMRSS;
       vftr_fp_vmrss = fopen ("/proc/self/status", "r");
       vftr_memtrace = true;
     } else {
       vftr_memtrace = false;
     }
   }

}

/**********************************************************************/

void vftr_finalize_mallinfo() {
   if (vftr_fp_vmrss != NULL) fclose (vftr_fp_vmrss);
   vftr_fp_vmrss = NULL;
   vftr_memtrace = false;
}

/**********************************************************************/

void vftr_process_mallinfo_line(char *line, long *count, long long *size) {
   char *tmp = strtok(line, "=\"");
   tmp = strtok(NULL, "\"");
   tmp = strtok(NULL, "=\"");
   tmp = strtok(NULL, "\"");
   *count = atol(tmp);
   tmp = strtok(NULL, "=\"");
   tmp = strtok(NULL, "\"");
   *size = atol(tmp);
}

/**********************************************************************/

long long vftr_get_vmrss(bool verbose) {
   long long vmrss = 0;
   char line[1024];
   while (fgets(line, 1024, vftr_fp_vmrss)) {
      if (strstr(line, "VmRSS:") != NULL) {
         if (verbose) printf ("VFTRACE: %s\n", line);
         char *tmp = strtok(line, "\t");
         tmp = strtok (NULL, " ");
         vmrss = atol(tmp);
      }
   }
   rewind(vftr_fp_vmrss);
   
   return vmrss;
}

/**********************************************************************/

void vftr_sample_vmrss (long long n_calls, bool is_entry, bool verbose, mem_prof_t *mem_prof) {
   vftr_mallinfo_ovhd -= vftr_get_runtime_usec();
   long long next_memtrace = is_entry ? mem_prof->next_memtrace_entry : mem_prof->next_memtrace_exit;
   if (n_calls >= next_memtrace) {
      long long vmrss = vftr_get_vmrss (verbose);
      long long tmp = is_entry ? mem_prof->mem_exit : mem_prof->mem_entry;
      bool needs_increment = n_calls > 0 &&
          (llabs(tmp - vmrss) == 0 || llabs(mem_prof->mem_exit - mem_prof->mem_entry - vmrss) < mem_prof->mem_tolerance);
      if (needs_increment) {
         if (is_entry) {
            mem_prof->next_memtrace_entry += mem_prof->mem_increment;
         } else {
            mem_prof->next_memtrace_exit += mem_prof->mem_increment;
         }
      }
      if (is_entry) {
         mem_prof->mem_entry = vmrss;
      } else { // is_exit
         mem_prof->mem_exit = vmrss;
         if (mem_prof->mem_exit - mem_prof->mem_entry > mem_prof->mem_max) mem_prof->mem_max = mem_prof->mem_exit - mem_prof->mem_entry;
      }
   }
   vftr_mallinfo_ovhd += vftr_get_runtime_usec();
}

/**********************************************************************/


void vftr_get_mallinfo () {
   char *buf;
   size_t bufsize;
   vftr_mallinfo_ovhd -= vftr_get_runtime_usec();
   FILE *fp = open_memstream (&buf, &bufsize);  
   malloc_info (0, fp);
   fclose(fp);
   vftr_mallinfo_ovhd += vftr_get_runtime_usec();
 
   vftr_mallinfo_post_ovhd -= vftr_get_runtime_usec();
   char *token = strtok(buf, "\n"); 
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

/**********************************************************************/
