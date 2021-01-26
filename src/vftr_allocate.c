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
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "vftr_setup.h"

typedef struct allocate_list {
   char *name;
   char *caller;
   int n_calls;
   long long allocated_memory;
   uint64_t id;
   bool open;
   bool need_warning;
} allocate_list_t;

#define INIT_ALLOC_LIST 1000
allocate_list_t *vftr_allocated_fields[1000];
int vftr_n_allocated_fields = 0;
int vftr_max_allocated_fields = 0;

void vftr_allocate_new_field (const char *name) {
   //vftr_allocated_fields[vftr_n_allocated_fields] = (allocate_list_t*) malloc (sizeof(allocate_list_t));
   allocate_list_t *new_field = (allocate_list_t*) malloc (sizeof(allocate_list_t));
   new_field->name = strdup(name);
   new_field->caller = NULL;
   new_field->n_calls = 0;
   new_field->allocated_memory = 0;
   new_field->id = vftr_jenkins_murmur_64_hash (strlen(name), (uint8_t*)name);
   //if (vftr_mpirank == 0) printf ("THIS ID: %lld\n", new_field->id);
   new_field->open = true;
   vftr_allocated_fields[vftr_max_allocated_fields++] = new_field;
}

int vftr_allocate_find_field (const char *name) {
   uint64_t this_id = vftr_jenkins_murmur_64_hash (strlen(name), (uint8_t*)name);
   for (int i = vftr_max_allocated_fields - 1; i >=0; i--) {
      //if (vftr_mpirank == 0) printf ("Compare IDs: %llu %llu\n", this_id, vftr_allocated_fields[i]->id);
      if (this_id == vftr_allocated_fields[i]->id) return i;
   }
   return -1;
}

void vftr_allocate_count (int index, long long alloc_size) {
   vftr_allocated_fields[index]->n_calls++;
   vftr_allocated_fields[index]->allocated_memory += alloc_size;
}

void vftr_allocate_set_open_state (int index) {
   if (vftr_allocated_fields[index]->open) {
      vftr_allocated_fields[index]->need_warning = true;
   } else {
      vftr_allocated_fields[index]->open = true;
   }
}

//void vftrace_allocate (const char *s, const int *dims, const int *n, const int *element_size) {
void vftrace_allocate (const char *s, const int *n_elements, const int *element_size) {
   //if (vftr_mpirank == 0) {
   //   printf ("REGISTER ALLOCATE: %s ", s);
   //   for (int i = 0; i < *n; i++) {
   //     printf ("%d ", dims[i]);
   //   }
   //   printf ("\n");
   //}
   //vftr_allocate_new_field (s);
   int index = vftr_allocate_find_field (s);
   //if (vftr_mpirank == 0) printf ("Already there? %s\n", index >= 0 ? "YES" : "NO");
   if (index < 0) {
      vftr_allocate_new_field (s);
      index = vftr_max_allocated_fields - 1;
   } else {
      vftr_allocate_set_open_state (index);
   }
   long long this_alloc = (long long)((*element_size) * (*n_elements));
   //for (int i = 0; i < *n; i++) {
   //  this_alloc *= dims[i];
   //}
   vftr_allocate_count (index, this_alloc);
   vftr_n_allocated_fields++; 
   
   //printf ("REGISTER: %s %d %d %d\n", s, *n, dims[0], dims[1]);
}

void vftrace_deallocate (const char *s) {
   //if (vftr_mpirank == 0) printf ("CALL VFTRACE_DEALLOCATE: %s\n", s);
   int index = vftr_allocate_find_field (s);
   vftr_allocated_fields[index]->open = false;
   vftr_n_allocated_fields--;
}

void vftr_allocate_finalize () {
   if (vftr_mpirank == 0) {
      printf ("Vftrace memory allocation report:\n");
      printf ("Registered fields: %d\n", vftr_max_allocated_fields);
      printf ("Unresolved allocations: %d\n", vftr_n_allocated_fields);
      printf ("***************************************************\n");
      printf ("%s %s %s %s\n", "Field name", "MB total", "calls", "MB / call");
      for (int i = 0; i < vftr_max_allocated_fields; i++) {
        double mb = (double)vftr_allocated_fields[i]->allocated_memory / 1024 / 1024;
        printf ("%s %lf %d %lf\n", vftr_allocated_fields[i]->name, mb, vftr_allocated_fields[i]->n_calls, mb / vftr_allocated_fields[i]->n_calls);
      }
   }
}





