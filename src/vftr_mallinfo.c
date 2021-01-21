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
#include "vftr_mallinfo.h"

vftr_mallinfo_t vftr_current_mallinfo;
bool vftr_memtrace;

void vftr_init_mallinfo () {
   memset (&vftr_current_mallinfo, 0, sizeof(vftr_mallinfo_t));
   vftr_memtrace = true;
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

void vftr_get_mallinfo () {
   //static int first = 1;
   //if (!first) return;
   char *buf;
   size_t bufsize;
   FILE *fp = open_memstream (&buf, &bufsize);  
   malloc_info (0, fp);
   fclose(fp);
 
   char *lines[VFTR_XML_STRING_LENGTH];
   char *token = strtok(buf, "\n"); 
   int i = 0;
   while (token != NULL) {
      //if (vftr_mpirank == 0) printf ("i: %d, token: %s\n", i, token);
      lines[i++] = token; 
      token = strtok(NULL, "\n");
   }
   //for (int rank = 0; i < vftr_mpisize; i++) {
   //for (i = 0; i < VFTR_XML_STRING_LENGTH; i++) {
   //  if (vftr_mpirank == 0) printf ("%d: %s\n", i, lines[i]);
   //}
   //PMPI_Barrier(MPI_COMM_WORLD);
   //}
   vftr_process_mallinfo_line (lines[MEM_MMAP], &(vftr_current_mallinfo.mmap_count), &(vftr_current_mallinfo.mmap_size));
   //if (vftr_mpirank == 0) printf ("Check: %ld %ld\n", vftr_current_mallinfo.mmap_count, vftr_current_mallinfo.mmap_size);
   free(buf);
   //first = 0;
}
