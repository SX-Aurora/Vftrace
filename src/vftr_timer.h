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

#ifndef VFTR_TIMER_H
#define VFTR_TIMER_H

#include <time.h>

typedef struct CallsTime {
    long long calls;
    float     time;
} callsTime_t;

typedef struct CallsTimeRange {
    int       stackIndex;
    long long minMPIcalls,maxMPIcalls,avgMPIcalls,
              minOMPcalls,maxOMPcalls,avgOMPcalls;
    float     minMPItime, maxMPItime, avgMPItime,
              minOMPtime, maxOMPtime, avgOMPtime;
    int       minMPIindxc,maxMPIindxc,minOMPindxc,maxOMPindxc,
              minMPIindxt,maxMPIindxt,minOMPindxt,maxOMPindxt;
} callsTimeRange_t;

// global sample timer
long long vftr_prevsampletime;
long long vftr_nextsampletime;

//sample time in ms
extern long long vftr_interval;
// maximum runtime in ms
extern long long vftr_timelimit;

// set the local reference time to which all 
// timedifferences are measured
void vftr_set_local_ref_time();

// get the current time in micro seconds since
// the reference time point
long long vftr_get_runtime_usec();
// get the elapsed number of VE cycles
unsigned long long vftr_get_cycles();

// Vftrace measures its own overhead in microseconds. 
// The array is allocated to the number of OpenMP threads
// It is incremented at each function entry and exit, as
// well as after initialization.
long long vftr_overhead_usec;

// The timestamp of initialization, set after call to MPI_Init().
long long vftr_inittime;
// What is this for?
long long vftr_inittime1;

long long vftr_initcycles;

// A time interval indicating when the function table should be sorted.
// This is done also to dynamically assign the "detail" flag if the weight
// of a function (in terms of cumulative cycles) grows. 
double vftr_sorttime;

// After each sorting, the vftr_sorttime is multiplied by this factor
double vftr_sorttime_growth;

#endif
