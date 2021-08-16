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
#include "vftr_environment.h"
#include "vftr_filewrite.h"
#include "vftr_stacks.h"
#include "vftr_pause.h"
#include "vftr_allocate.h"
#include "vftr_hashing.h"

typedef struct allocate_list {
   char *name;
   char *caller;
   int n_calls;
   int stack_id;
   long long max_memory;
   long long global_max;
   long long allocated_memory;
   uint64_t id;
   bool open;
   bool need_warning;
} allocate_list_t;

#define INIT_ALLOC_LIST 1000
#define ALLOC_LIST_INC 500
allocate_list_t **vftr_allocated_fields;
int vftr_allocate_list_size = 0;
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

int vftr_compare_max_memory (const void *a1, const void *a2) {
  allocate_list_t *l1 = *(allocate_list_t **)a1;
  allocate_list_t *l2 = *(allocate_list_t **)a2;
  if (!l1) return -1;
  if (!l2) return 1;
  long long diff = l2->global_max - l1->global_max;
  if (diff > 0) return 1;
  if (diff < 0) return -1;
  return 0;
}

/**********************************************************************/

void vftr_allocate_new_field (const char *name, const char *caller_function, int stack_id) {
   allocate_list_t *new_field = (allocate_list_t*) malloc (sizeof(allocate_list_t));
   new_field->name = strdup(name);
   new_field->caller = strdup(caller_function);
   new_field->n_calls = 0;
   new_field->stack_id = stack_id;
   new_field->allocated_memory = 0;
   new_field->max_memory = 0;
   new_field->global_max = 0;
   char name_and_caller[strlen(name) + strlen(caller_function) + 1];
   snprintf (name_and_caller, strlen(name) + strlen(caller_function) + 1, "%s%s", name, caller_function);
   new_field->id = vftr_jenkins_murmur_64_hash (strlen(name_and_caller), (uint8_t*)name_and_caller);
   new_field->open = true;
   if (vftr_max_allocated_fields + 1 > vftr_allocate_list_size) {
      vftr_allocate_list_size += ALLOC_LIST_INC;
      vftr_allocated_fields = (allocate_list_t**)realloc (vftr_allocated_fields, vftr_allocate_list_size * sizeof(allocate_list_t*));
   }
   vftr_allocated_fields[vftr_max_allocated_fields++] = new_field;
}

/**********************************************************************/

int vftr_allocate_find_field (const char *name, const char *caller_function) {
   char name_and_caller[strlen(name) + strlen(caller_function) + 1];
   snprintf (name_and_caller, strlen(name) + strlen(caller_function) + 1, "%s%s", name, caller_function);
   //uint64_t this_id = vftr_jenkins_murmur_64_hash (strlen(name), (uint8_t*)name);
   uint64_t this_id = vftr_jenkins_murmur_64_hash (strlen(name_and_caller), (uint8_t*)name_and_caller);
   for (int i = vftr_max_allocated_fields - 1; i >=0; i--) {
      if (this_id == vftr_allocated_fields[i]->id) return i;
   }
   return -1;
}

/**********************************************************************/

void vftr_allocate_count (int index, long long alloc_size) {
   vftr_allocated_fields[index]->n_calls++;
   vftr_allocated_fields[index]->allocated_memory += alloc_size;
   if (alloc_size > vftr_allocated_fields[index]->max_memory) vftr_allocated_fields[index]->max_memory = alloc_size;
}

/**********************************************************************/

void vftr_allocate_get_memory_for_stackid (int stack_id, long long *mem_tot, long long *mem_max) {
//long long vftr_allocate_get_max_memory_for_stackid (int stack_id) {
   //long long mem_out = 0; 
   *mem_tot = 0;
   *mem_max = 0;
   for (int i = 0; i < vftr_max_allocated_fields; i++) {
      if (vftr_allocated_fields[i]->stack_id == stack_id) {
        if (vftr_allocated_fields[i]->max_memory > *mem_max) *mem_max = vftr_allocated_fields[i]->max_memory;
        *mem_tot += vftr_allocated_fields[i]->allocated_memory;
      }
   }
   //return mem_out;
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

void vftrace_allocate (const char *s, const int *n_elements, const int *element_size) {
   if (vftr_off() || vftr_paused || vftr_env_no_memtrace()) return;
   if (vftr_allocate_list_size == 0) {
      vftr_allocate_list_size = INIT_ALLOC_LIST;
      vftr_allocated_fields = (allocate_list_t**)malloc (vftr_allocate_list_size * sizeof(allocate_list_t*));
   }
   int index = vftr_allocate_find_field (s, vftr_fstack->name);
   if (index < 0) {
      vftr_allocate_new_field (s, vftr_fstack->name, vftr_fstack->id);
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
   if (vftr_off() || vftr_paused || vftr_env_no_memtrace()) return;
   int index = vftr_allocate_find_field (s, vftr_fstack->name);
   vftr_allocated_fields[index]->open = false;
   vftr_n_allocated_fields--;
}

/**********************************************************************/

void vftr_allocate_finalize (FILE *fp) {

   if (vftr_mpisize > 1) {
#ifdef _MPI
     // Search all the ranks for global maximal values.  
     PMPI_Barrier (MPI_COMM_WORLD);
     int all_n_allocated[vftr_mpisize];
     if (vftr_mpirank > 0) {
        PMPI_Send (&vftr_max_allocated_fields, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
     } else {
        all_n_allocated[0] = vftr_max_allocated_fields;
        for (int i = 1; i < vftr_mpisize; i++) {
           int tmp;
           PMPI_Recv(&tmp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
           all_n_allocated[i] = tmp;
        }
     }
     uint64_t *all_hashes[vftr_mpisize];
     if (vftr_mpirank == 0) {
        for (int i = 0; i < vftr_mpisize; i++) {
           all_hashes[i] = (uint64_t*)malloc (all_n_allocated[i] * sizeof(uint64_t));
        }
     }
     uint64_t *my_hashes = (uint64_t*)malloc (vftr_max_allocated_fields * sizeof(uint64_t));
     for (int i = 0; i < vftr_max_allocated_fields; i++) {
       my_hashes[i] = vftr_allocated_fields[i]->id;
       if (vftr_mpirank == 0) all_hashes[0][i] = my_hashes[i];
     }
     if (vftr_mpirank > 0) {
       PMPI_Send (my_hashes, vftr_max_allocated_fields, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD);
     } else {
       for (int i = 1; i < vftr_mpisize; i++) {
          PMPI_Recv(all_hashes[i], all_n_allocated[i], MPI_UINT64_T, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       }
     }
  
     long long *all_max_memories[vftr_mpisize];
     if (vftr_mpirank == 0) {
       for (int i = 0; i < vftr_mpisize; i++) {
          all_max_memories[i] = (long long *)malloc (all_n_allocated[i] * sizeof(long long));
       }
     }
     long long *my_max_mem = (long long*)malloc (vftr_max_allocated_fields * sizeof(long long));
     for (int i = 0; i < vftr_max_allocated_fields; i++) {
        my_max_mem[i] = vftr_allocated_fields[i]->max_memory;
        if (vftr_mpirank == 0) all_max_memories[0][i] = my_max_mem[i];
     }
     if (vftr_mpirank > 0) {
        PMPI_Send (my_max_mem, vftr_max_allocated_fields, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
     } else {
       for (int i = 1; i < vftr_mpisize; i++) {
          PMPI_Recv(all_max_memories[i], all_n_allocated[i], MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
       }
     }
  
     int total_allocated = 0;
     for (int i = 0; i < vftr_mpisize; i++) {
        total_allocated += all_n_allocated[i];
     }
     uint64_t *global_hashes;
     long long *global_max;
     int n_unique_hashes;
  
     if (vftr_mpirank == 0) {
        bool already_there;
        n_unique_hashes = 0;
        uint64_t *tmp_hashes = (uint64_t*)malloc (total_allocated * sizeof(uint64_t));
        for (int rank = 0; rank < vftr_mpisize; rank++) {
           for (int i = 0; i < all_n_allocated[rank]; i++) { 
              already_there = false;
              for (int j = 0; j < n_unique_hashes; j++) {
                 if (tmp_hashes[j] == all_hashes[rank][i]) {
                    already_there = true;
                    break;
                 }
              }
              if (!already_there) {
                  tmp_hashes[n_unique_hashes++] = all_hashes[rank][i];
              }
           }
        } 
        global_hashes = (uint64_t*)malloc (n_unique_hashes * sizeof(uint64_t));
  
        int has_hash[n_unique_hashes][vftr_mpisize];
        for (int i = 0; i < n_unique_hashes; i++) {
          global_hashes[i] = tmp_hashes[i];
          for (int rank = 0; rank < vftr_mpisize; rank++) {
             has_hash[i][rank] = -1;
             for (int i_local = 0; i_local < all_n_allocated[rank]; i_local++) {
                if (all_hashes[rank][i_local] == global_hashes[i]) {
                  has_hash[i][rank] = i_local;
                  break;
                }
             }
           }
        }
        global_max = (long long*) malloc (n_unique_hashes * sizeof(long long));
        for (int i = 0; i < n_unique_hashes; i++) {
           global_max[i] = 0;
           for (int rank = 0; rank < vftr_mpisize; rank++) {
              int i_local = has_hash[i][rank];
              if (i_local > -1) {
                if (all_max_memories[rank][i_local] > global_max[i]) global_max[i] = all_max_memories[rank][i_local];
              }
           }
        }
        for (int i = 1; i < vftr_mpisize; i++) {
           PMPI_Send (&n_unique_hashes, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
           PMPI_Send (global_hashes, n_unique_hashes, MPI_UINT64_T, i, 0, MPI_COMM_WORLD);
           PMPI_Send (global_max, n_unique_hashes, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD);
        }
        free(tmp_hashes);
     } else {
        PMPI_Recv (&n_unique_hashes, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        global_hashes = (uint64_t*)malloc(n_unique_hashes * sizeof(uint64_t));
        global_max = (long long*)malloc(n_unique_hashes * sizeof(long long));
        PMPI_Recv (global_hashes, n_unique_hashes, MPI_UINT64_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        PMPI_Recv (global_max, n_unique_hashes, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
     }
  
     for (int i = 0; i < vftr_max_allocated_fields; i++) {
       uint64_t this_hash = vftr_allocated_fields[i]->id;
       for (int j = 0; j < n_unique_hashes; j++) {
         if (this_hash == global_hashes[j]) {
           vftr_allocated_fields[i]->global_max = global_max[j];
           break;
         }
       }
     }
#endif
  } else {
    vftr_allocated_fields[0]->global_max = vftr_allocated_fields[0]->max_memory;   
  }

   qsort ((void*)vftr_allocated_fields, (size_t)vftr_max_allocated_fields,
          sizeof(allocate_list_t **), vftr_compare_max_memory); 
 
   column_t columns[7];
   vftr_prof_column_init ("Field name", NULL, 0, COL_CHAR_RIGHT, SEP_MID, &columns[0]);
   vftr_prof_column_init ("Called by", NULL, 0, COL_CHAR_RIGHT, SEP_MID, &columns[1]);
   vftr_prof_column_init ("Total memory", NULL, 2, COL_MEM, SEP_MID, &columns[2]);
   vftr_prof_column_init ("n_calls", NULL, 0, COL_INT, SEP_MID, &columns[3]);
   vftr_prof_column_init ("Memory / call", NULL, 2, COL_MEM, SEP_MID, &columns[4]);
   vftr_prof_column_init ("Global Max", NULL, 2, COL_MEM, SEP_MID, &columns[5]);
   vftr_prof_column_init ("ID", NULL, 0, COL_INT, SEP_LAST, &columns[6]);
   for (int i = 0; i < vftr_max_allocated_fields; i++) {
     double mb = (double)vftr_allocated_fields[i]->allocated_memory;
     double mb_per_call = mb / vftr_allocated_fields[i]->n_calls;
     int id = vftr_func_table[vftr_allocated_fields[i]->stack_id]->gid;
     int stat;
     vftr_prof_column_set_n_chars (vftr_allocated_fields[i]->name, NULL, NULL, &columns[0], &stat);
     vftr_prof_column_set_n_chars (vftr_allocated_fields[i]->caller, NULL, NULL, &columns[1], &stat);
     vftr_prof_column_set_n_chars (&mb, NULL, NULL, &columns[2], &stat);
     vftr_prof_column_set_n_chars (&vftr_allocated_fields[i]->n_calls, NULL, NULL, &columns[3], &stat);
     vftr_prof_column_set_n_chars (&mb_per_call, NULL, NULL, &columns[4], &stat);
     double gm = (double)vftr_allocated_fields[i]->global_max;
     vftr_prof_column_set_n_chars (&gm, NULL, NULL, &columns[5], &stat);
     vftr_prof_column_set_n_chars (&id, NULL, NULL, &columns[6], &stat);
   }
   fprintf (fp, "Vftrace memory allocation report:\n");
   fprintf (fp, "Registered fields: %d\n", vftr_max_allocated_fields);
   fprintf (fp, "***************************************************\n");
   fprintf (fp, "| %*s | %*s | %*s | %*s | %*s | %*s | %*s |\n",
            columns[0].n_chars, columns[0].header,
            columns[1].n_chars, columns[1].header,
            columns[2].n_chars, columns[2].header,
            columns[3].n_chars, columns[3].header,
            columns[4].n_chars, columns[4].header,
            columns[5].n_chars, columns[5].header,
            columns[6].n_chars, columns[6].header);
   int table_width = vftr_get_tablewidth_from_columns (columns, 6, true);
   for (int i = 0; i < table_width; i++) fprintf (fp, "-");
   fprintf (fp, "\n");
   for (int i = 0; i < vftr_max_allocated_fields; i++) {
     double mb = (double)vftr_allocated_fields[i]->allocated_memory;
     double mb_per_call = mb / vftr_allocated_fields[i]->n_calls;
     vftr_prof_column_print (fp, columns[0], vftr_allocated_fields[i]->name, NULL, NULL);
     vftr_prof_column_print (fp, columns[1], vftr_allocated_fields[i]->caller, NULL, NULL);
     vftr_prof_column_print (fp, columns[2], &mb, NULL, NULL);
     vftr_prof_column_print (fp, columns[3], &vftr_allocated_fields[i]->n_calls, NULL, NULL);
     vftr_prof_column_print (fp, columns[4], &mb_per_call, NULL, NULL);
     double gm = (double)vftr_allocated_fields[i]->global_max;
     vftr_prof_column_print (fp, columns[5], &gm, NULL, NULL);
     int id = vftr_func_table[vftr_allocated_fields[i]->stack_id]->gid;
     vftr_prof_column_print (fp, columns[6], &id, NULL, NULL);
     fprintf (fp, "\n");
   }
   for (int i = 0; i < table_width; i++) fprintf (fp, "-");
   fprintf (fp, "\n");
}

/**********************************************************************/



