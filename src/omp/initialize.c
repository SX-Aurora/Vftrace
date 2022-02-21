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

#include <omp.h>
#include <omp-tools.h>

#include "vftr_timer.h"
#include "omp_timer.h"

#include "parallel_begin.h"
#include "parallel_end.h"
#include "thread_begin.h"
#include "thread_end.h"
#include "sync_region.h"
#include "sync_region_wait.h"

int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num,
                    ompt_data_t *tool_data) {
   // Start counter for overhead
   long long tstart = vftr_get_runtime_usec();

   // Get the set_callback function pointer
   ompt_set_callback_t ompt_set_callback =
      (ompt_set_callback_t)lookup("ompt_set_callback");
   // register the available callback functions
   vftr_register_ompt_callback_parallel_begin(ompt_set_callback);
   vftr_register_ompt_callback_parallel_end(ompt_set_callback);
//   vftr_register_ompt_callback_thread_begin(ompt_set_callback);
//   vftr_register_ompt_callback_thread_end(ompt_set_callback);
   vftr_register_ompt_callback_sync_region(ompt_set_callback);
//   vftr_register_ompt_callback_sync_region_wait(ompt_set_callback);

   long long tend = vftr_get_runtime_usec();
   vftr_omp_overhead_usec += tend - tstart;
   return 1; // success: activates tool
}

// define the function pointer 
int (*ompt_initialize_ptr)(ompt_function_lookup_t, int, ompt_data_t*) = ompt_initialize;
