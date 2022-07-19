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
#include "vftr_initialize.h"

// main datatype to store everything

vftrace_t vftrace = {
   .hooks = {
      .function_hooks = {
         .enter = &vftr_initialize,
         .exit = NULL
      }
   },
   .environment.valid = false,
   .symboltable = {
      .nsymbols = 0,
      .symbols = NULL
   },
   .process = {
      .nprocesses = 1,
      .processID = 0,
      .stacktree = {
         .nstacks = 0,
         .maxstacks = 0,
         .stacks = NULL
      },
      .collated_stacktree = {
         .nstacks = 0,
         .stacks = 0
      }
   },
   .state = undefined,
   .sampling = {
      .do_sampling = false,
      .vfdfilename = NULL,
      .vfdfilefp = NULL,
      .iobuffer = NULL,
      .nextsampletime = 0,
      .interval = 0,
      .function_samplecount = 0,
      .message_samplecount = 0,
      .stacktable_offset = 0,
      .samples_offset = 0
   },
#ifdef _OMP
   .omp_state = {
      .tool_started = false,
      .initialized = false,
      .omp_version = 0,
      .runtime_version = NULL,
   },
#endif
#ifdef _MPI
   .mpi_state = {
      .pcontrol_level = 1,
      .nopen_requests = 0,
      .open_requests = NULL,
      .nprof_ranks = 0,
      .prof_ranks = NULL,
      .my_rank_in_prof = false
   },
#endif
   .timestrings = {
      .start_time = NULL,
      .end_time = NULL
   }
};
