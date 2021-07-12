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

#ifndef VFTR_ALLOCATE_H
#define VFTR_ALLOCATE_H

extern int vftr_max_allocated_fields;

//long long vftr_allocate_get_max_memory_for_stackid (int stack_id);
void vftr_allocate_get_memory_for_stackid (int stack_id, long long *mem_tot, long long *mem_max);

void vftrace_allocate (const char *s, const int *n_elements, const int *element_size);
void vftrace_deallocate (const char *s);

void vftr_allocate_finalize(FILE *fp);

#endif
