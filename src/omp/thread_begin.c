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

#include <stdio.h>

#include <omp.h>
#include <omp-tools.h>

static void vftr_ompt_callback_thread_begin(ompt_thread_t thread_type,
                                            ompt_data_t *thread_data) {
   printf("starting thread\n");
}

void vftr_register_ompt_callback_thread_begin(ompt_set_callback_t ompt_set_callback) {
   ompt_callback_thread_begin_t f_ompt_callback_thread_begin =
      &vftr_ompt_callback_thread_begin;
   if (ompt_set_callback(ompt_callback_thread_begin, (ompt_callback_t)f_ompt_callback_thread_begin) == ompt_set_never) {
      fprintf(stderr, "0: Could not register callback \"ompt_callback_thread_begin\"\n");
   }
}
