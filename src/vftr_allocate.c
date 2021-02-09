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
#include "vftr_filewrite.h"
#include "vftr_stacks.h"

typedef struct allocate_list {
   char *name;
   char *caller;
   int n_calls;
   long long allocated_memory;
   uint64_t id;
   bool open;
   bool need_warning;
} allocate_list_t;

#define INIT_ALLOC_LIST 10000
allocate_list_t *vftr_allocated_fields[INIT_ALLOC_LIST];
int vftr_n_allocated_fields = 0;
int vftr_max_allocated_fields = 0;

/**********************************************************************/

int vftr_compare_allocated_memory (const void *a1, const void *a2) {
  allocate_list_t *l1 = *(allocate_list_t **)a1;
  allocate_list_t *l2 = *(allocate_list_t **)a2;
  if (!l1) return -1;
  if (!l2) return 1;
  long long diff = l2->allocated_memory - l1->allocated_memory;
  if (diff > 0) return 1;
  if (diff < 0) return -1;
  return 0;
}

/**********************************************************************/

void vftr_allocate_new_field (const char *name, const char *caller_function) {
   allocate_list_t *new_field = (allocate_list_t*) malloc (sizeof(allocate_list_t));
   new_field->name = strdup(name);
   new_field->caller = strdup(caller_function);
   new_field->n_calls = 0;
   new_field->allocated_memory = 0;
   char name_and_caller[strlen(name) + strlen(caller_function) + 1];
   snprintf (name_and_caller, strlen(name) + strlen(caller_function) + 1, "%s%s", name, caller_function);
   new_field->id = vftr_jenkins_murmur_64_hash (strlen(name_and_caller), (uint8_t*)name_and_caller);
   new_field->open = true;
   vftr_allocated_fields[vftr_max_allocated_fields++] = new_field;
}

/**********************************************************************/

int vftr_allocate_find_field (const char *name, const char *caller_function) {
   uint64_t this_id = vftr_jenkins_murmur_64_hash (strlen(name), (uint8_t*)name);
   for (int i = vftr_max_allocated_fields - 1; i >=0; i--) {
      if (this_id == vftr_allocated_fields[i]->id) return i;
   }
   return -1;
}

/**********************************************************************/

void vftr_allocate_count (int index, long long alloc_size) {
   vftr_allocated_fields[index]->n_calls++;
   vftr_allocated_fields[index]->allocated_memory += alloc_size;
}

/**********************************************************************/

void vftr_allocate_set_open_state (int index) {
   if (vftr_allocated_fields[index]->open) {
      vftr_allocated_fields[index]->need_warning = true;
   } else {
      vftr_allocated_fields[index]->open = true;
   }
}

/**********************************************************************/

//void vftrace_allocate (const char *s, const int *dims, const int *n, const int *element_size) {
void vftrace_allocate (const char *s, const int *n_elements, const int *element_size) {
   int index = vftr_allocate_find_field (s, vftr_fstack->name);
   if (index < 0) {
      vftr_allocate_new_field (s, vftr_fstack->name);
      index = vftr_max_allocated_fields - 1;
   } else {
      vftr_allocate_set_open_state (index);
   }
   long long this_alloc = (long long)((*element_size) * (*n_elements));
   vftr_allocate_count (index, this_alloc);
   vftr_n_allocated_fields++; 
   
}

/**********************************************************************/

void vftrace_deallocate (const char *s) {
   //if (vftr_mpirank == 0) printf ("CALL VFTRACE_DEALLOCATE: %s\n", s);
   int index = vftr_allocate_find_field (s, vftr_fstack->name);
   vftr_allocated_fields[index]->open = false;
   vftr_n_allocated_fields--;
}

/**********************************************************************/

void vftr_allocate_finalize () {
   if (vftr_mpirank == 0) {
      qsort ((void*)vftr_allocated_fields, (size_t)vftr_max_allocated_fields,
             sizeof(allocate_list_t *), vftr_compare_allocated_memory); 
      column_t columns[5];
      vftr_prof_column_init ("Field name", NULL, 0, COL_CHAR, SEP_MID, &columns[0]);
      vftr_prof_column_init ("Called by", NULL, 0, COL_CHAR, SEP_MID, &columns[1]);
      vftr_prof_column_init ("Total memory", NULL, 0, COL_MEM, SEP_MID, &columns[2]);
      vftr_prof_column_init ("n_calls", NULL, 0, COL_INT, SEP_MID, &columns[3]);
      vftr_prof_column_init ("Memory / call", NULL, 0, COL_MEM, SEP_MID, &columns[4]);
      for (int i = 0; i < vftr_max_allocated_fields; i++) {
        //double mb = (double)vftr_allocated_fields[i]->allocated_memory / 1024 / 1024;
        double mb = (double)vftr_allocated_fields[i]->allocated_memory;
        double mb_per_call = mb / vftr_allocated_fields[i]->n_calls;
        int stat;
        vftr_prof_column_set_n_chars (vftr_allocated_fields[i]->name, NULL, &columns[0], &stat);
        vftr_prof_column_set_n_chars (vftr_allocated_fields[i]->caller, NULL, &columns[1], &stat);
        vftr_prof_column_set_n_chars (&mb, NULL, &columns[2], &stat);
        vftr_prof_column_set_n_chars (&vftr_allocated_fields[i]->n_calls, NULL, &columns[3], &stat);
        vftr_prof_column_set_n_chars (&mb_per_call, NULL, &columns[4], &stat);
      }
      fprintf (stdout, "Vftrace memory allocation report:\n");
      fprintf (stdout, "Registered fields: %d\n", vftr_max_allocated_fields);
      fprintf (stdout, "Unresolved allocations: %d\n", vftr_n_allocated_fields);
      fprintf (stdout, "***************************************************\n");
      for (int i = 0; i < 5; i++) {
        fprintf (stdout, " %*s ", columns[i].n_chars, columns[i].header);
      }
      fprintf (stdout, "\n");
      for (int i = 0; i < vftr_max_allocated_fields; i++) {
        double mb = (double)vftr_allocated_fields[i]->allocated_memory;
        double mb_per_call = mb / vftr_allocated_fields[i]->n_calls;
        vftr_prof_column_print (stdout, columns[0], vftr_allocated_fields[i]->name, NULL);
        vftr_prof_column_print (stdout, columns[1], vftr_allocated_fields[i]->caller, NULL);
        vftr_prof_column_print (stdout, columns[2], &mb, NULL);
        vftr_prof_column_print (stdout, columns[3], &vftr_allocated_fields[i]->n_calls, NULL);
        vftr_prof_column_print (stdout, columns[4], &mb_per_call, NULL);
        fprintf (stdout, "\n");
      }
      //printf ("%s %s %s %s %s\n", "Field name", "Called by", "MB total", "calls", "MB / call");
      //for (int i = 0; i < vftr_max_allocated_fields; i++) {
      //  double mb = (double)vftr_allocated_fields[i]->allocated_memory / 1024 / 1024;
      //  printf ("%s %s %lf %d %lf\n", vftr_allocated_fields[i]->name, vftr_allocated_fields[i]->caller, 
      //          mb, vftr_allocated_fields[i]->n_calls, mb / vftr_allocated_fields[i]->n_calls);
      //}
   }
}

/**********************************************************************/



