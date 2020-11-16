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

#ifndef VFTR_SORTING_H
#define VFTR_SORTING_H

#include <stdint.h>
#include <stdbool.h>

#include "vftr_functions.h"

void vftr_radixsort_uint64(int n, uint64_t *list);

void vftr_sort_integer (int *i_array, int n, bool ascending);
void vftr_sort_double (double *d_array, int n, bool ascending);
void vftr_sort_double_copy (double *d_array, int n, bool ascending, double *d_copy);

int vftr_compare_function_excl_time (const void *a1, const void *a2);
int (*vftr_get_compare_function()) (const void *, const void *);

#endif
