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

#ifndef VFTRACE_H
#define VFTRACE_H

#ifdef __cplusplus 
extern "C" {
#endif
// Mark start of instrumented region 
// name ist the Region name to be used in the profile
void vftrace_region_begin(const char *name);

// Mark end of instrumented region 
// name ist the Region name to be used in the profile
void vftrace_region_end(const char *name);

void vftrace_allocate (const char *name, int n);

// obtain the stack string as char pointer
char *vftrace_get_stack();

// pause and resume sampling via vftrace in user code
void vftrace_pause();
void vftrace_resume();

void vftrace_show_callstack();
int vftrace_get_stacktree_size();

#ifdef __cplusplus 
}
#endif
#endif
