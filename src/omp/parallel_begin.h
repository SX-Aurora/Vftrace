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

#ifndef PARALLEL_BEGIN_H
#define PARALLEL_BEGIN_H

#include <omp.h>
#include <omp-tools.h>

static void vftr_ompt_callback_parallel_begin(ompt_data_t *encountering_task_data,
                                              const ompt_frame_t *encountering_task_frame,
                                              ompt_data_t *parallel_data,
                                              uint32_t requested_parallelism,
                                              int flags, const void *codeptr_ra);

void vftr_register_ompt_callback_parallel_begin(ompt_set_callback_t ompt_set_callback);

#endif
