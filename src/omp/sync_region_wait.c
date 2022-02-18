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

#include "vftr_timer.h"

static void vftr_ompt_callback_sync_region_wait(ompt_sync_region_t kind,
                                                ompt_scope_endpoint_t endpoint,
                                                ompt_data_t *parallel_data,
                                                ompt_data_t *task_data,
                                                const void *codeptr_ra) {
   switch (kind) {
      case ompt_sync_region_barrier:
         break;
      case ompt_sync_region_barrier_implicit:
         break;
      case ompt_sync_region_barrier_explicit:
         break;
      case ompt_sync_region_barrier_implementation:
         break;
      case ompt_sync_region_taskwait:
         break;
      case ompt_sync_region_taskgroup:
         break;
      case ompt_sync_region_reduction:
         break;
      case ompt_sync_region_barrier_implicit_workshare:
         break;
      case ompt_sync_region_barrier_implicit_parallel:
         printf("sync_region_wait id: %d (t: %lld)\n", omp_get_thread_num(), vftr_get_runtime_usec());
         break;
      case ompt_sync_region_barrier_teams:
         break;
      default:
         break;
   }
}

void vftr_register_ompt_callback_sync_region_wait(ompt_set_callback_t ompt_set_callback) {
   ompt_callback_sync_region_t f_ompt_callback_sync_region_wait =
      &vftr_ompt_callback_sync_region_wait;
   if (ompt_set_callback(ompt_callback_sync_region_wait, (ompt_callback_t)f_ompt_callback_sync_region_wait) == ompt_set_never) {
      fprintf(stderr, "0: Could not register callback \"ompt_callback_sync_region_wait\"\n");
   }
}
