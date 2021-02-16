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

#ifndef VFTR_REGIONS_H
#define VFTR_REGIONS_H

#include <stdbool.h>

void vftr_region_entry (const char *s, void *addr, bool isPrecise);
void vftr_region_exit();

// These regions are for vftrace internal usage only.
// They are always precise.
void vftr_internal_region_begin(const char *s);
void vftr_internal_region_end(const char *s);

void vftrace_allocate (const char *s, const int *dims, const int *n);

#endif
