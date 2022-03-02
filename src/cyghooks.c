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

#include "initialize.h"

// Define the function pointers which will be called by the hooks
// On the first function entry vftrace will be initialized.
// After that the function pointers will be redirected to the
// actual hook functionality or a dummy function if vftrace is off.
static void (*vftr_func_enter_hook_ptr)(void*, void*) = vftr_initialize;
static void (*vftr_func_exit_hook_ptr)(void*, void*) = NULL;

// Define functions to redirect the function hooks, so to not make the 
// function pointers globaly visible
void vftr_set_enter_func_hook(void (*function_ptr)(void*,void*)) {
   vftr_func_enter_hook_ptr = function_ptr;
}
void vftr_set_exit_func_hook(void (*function_ptr)(void*,void*)) {
   vftr_func_exit_hook_ptr = function_ptr;
}

#if defined(__x86_64__) || defined(__ve__)
void __cyg_profile_func_enter(void *func, void *call_site) {
   vftr_func_enter_hook_ptr(func, call_site);
}

void __cyg_profile_func_exit(void *func, void *call_site) {
   vftr_func_exit_hook_ptr(func, call_site);
}
#endif

#if defined(__ia64__)
// The argument func is a pointer to a pointer instead of a pointer.
void __cyg_profile_func_enter(void **func, void *call_site) {
   (void) call_site;
   vftr_function_entry(NULL, *func, false);
}

void __cyg_profile_func_exit(void **func, void *call_site) {
   vftr_func_exit_hook_ptr(*func, call_site);
}
#endif
