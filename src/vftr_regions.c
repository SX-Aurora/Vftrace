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

#include <stdbool.h>
#include "vftr_environment.h"
#include "vftr_hooks.h"
#include "vftr_pause.h"

void vftrace_region_begin(const char *s) {
    void *addr;
    if (vftr_off() || vftr_paused) return;
#ifdef __ve__
    asm volatile ("or %0,0,%%lr" : "=r" (addr));
#else
    asm volatile ("mov 8(%%rbp), %0" : "=r" (addr));
#endif
    bool precise = vftr_environment->regions_precise->value;
    vftr_function_entry(s, addr, 0, precise);
}

void vftrace_region_end(const char *s) {
    if(vftr_off() || vftr_paused) return;
    vftr_function_exit(0);
}
