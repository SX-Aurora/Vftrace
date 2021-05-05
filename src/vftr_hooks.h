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
#ifndef VFTR_HOOKS_H
#define VFTR_HOOKS_H

#include <stdbool.h>

extern bool vftr_profile_wanted;

// We keep a list of the addresses of the exluded functions.
// This way, we can check early in vftr_function_entry, if the
// given function should be skipped, before a time-consuming lookup in the
// function tree is done.
// current_exclude_addr stores the address of the last encountered excluded function.
// We assume, that consecutive calls are frequent, so this saves time by not needing to search in the list.
extern void *current_exclude_addr;
typedef struct excl_addr_list {
   void *addr;
   struct excl_addr_list *next;
} excl_fun_t;

extern excl_fun_t *exclude_addr;

void vftr_function_entry (const char *s, void *addr, bool isPrecise);
void vftr_function_exit ();

void vftr_save_old_state ();



#endif
