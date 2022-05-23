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

#ifndef VFTR_HWCOUNTERS_H
#define VFTR_HWCOUNTERS_H

#include <stdbool.h>

extern bool vftr_events_enabled;
extern int vftr_n_hw_obs;
extern long long vftr_prog_cycles;

#if defined( __ve__ )
#define MAX_HWC_EVENTS               16
#else
#define MAX_HWC_EVENTS                4
#endif
#define MAX_EVENTS                   (MAX_HWC_EVENTS+10)

#if defined(HAS_SXHWC)
#define VFTR_READ_COUNTERS vftr_read_counters_sx
#elif defined(HAS_PAPI)
#define VFTR_READ_COUNTERS vftr_read_counters_papi
#else
#define VFTR_READ_COUNTERS vftr_read_counters_dummy
#endif

extern long long vftr_echwc[MAX_HWC_EVENTS];

int vftr_init_hwc (char *scenario_file);
void vftr_init_hwc_memtrace();
int vftr_stop_hwc ();

void VFTR_READ_COUNTERS (long long *event);
void vftr_papi_counter (char *name);
void vftr_sx_counter (char *name, int id);

#if HAS_SXHWC
void vftr_read_sxhwc_registers (long long hwc[MAX_HWC_EVENTS]);
#endif

#endif
