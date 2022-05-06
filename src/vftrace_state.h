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

#ifndef VFTRACE_STATE_H
#define VFTRACE_STATE_H

#include <stdbool.h>

#include "hook_types.h"
#include "environment_types.h"
#include "symbol_types.h"
#include "process_types.h"
#include "sampling_types.h"
#include "timer_types.h"
#ifdef _OMP
#include "omp/omp_state_types.h"
#endif
#ifdef _MPI
#include "mpi/utils/mpi_state_types.h"
#endif

// tracing state of vftrace
typedef enum {
   undefined,
   off,
   on,
   paused
} state_t;

// main datatype to store everything
typedef struct {
   hooks_t hooks; // collection of function pointers
                  // where vftrace intercepts the program flow
   environment_t environment; // set of all relevant environment variables
   symboltable_t symboltable; // list of function symbols
   process_t process; // all of the dynamic process data
   state_t state; // current state of vftrace
   sampling_t sampling; // Filehandle and data required to handle vfd-file sampling
   time_strings_t timestrings; // start and end time in string form
#ifdef _OMP
   omp_state_t omp_state;
#endif
#ifdef _MPI
   mpi_state_t mpi_state;
#endif
} vftrace_t;

extern vftrace_t vftrace;

#endif
