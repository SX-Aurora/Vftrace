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

#include <stdlib.h>
#include <stdbool.h>

#include "vftrace_state.h"

#include "initialize.h"

// Define functions to redirect the function hooks, so to not make the 
// function pointers globaly visible
void vftr_set_enter_func_hook(void (*function_ptr)(void*,void*)) {
   vftrace.hooks.function_hooks.enter = function_ptr;
}
void vftr_set_exit_func_hook(void (*function_ptr)(void*,void*)) {
   vftrace.hooks.function_hooks.exit = function_ptr;
}

#if defined(__x86_64__) || defined(__ve__)
void __cyg_profile_func_enter(void *func, void *call_site) {
   vftrace.hooks.function_hooks.enter(func, call_site);
}

void __cyg_profile_func_exit(void *func, void *call_site) {
   vftrace.hooks.function_hooks.exit(func, call_site);
}
#endif

#if defined(__ia64__)
// The argument func is a pointer to a pointer instead of a pointer.
void __cyg_profile_func_enter(void **func, void *call_site) {
   (void) call_site;
   vftrace.hooks.function_hooks.enter(*func, call_site);
}

void __cyg_profile_func_exit(void **func, void *call_site) {
   vftrace.hooks.function_hooks.exit(*func, call_site);
}
#endif
