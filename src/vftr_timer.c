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

#include <time.h>

// global sample timer
long long vftr_prevsampletime;
long long vftr_nextsampletime;

//sample time in ms
long long vftr_interval;
// maximum runtime in ms
long long vftr_timelimit;

// the clock timer to use 
// CLOCK_MONOTONIC is not affected by NTP or system time changes.
const clockid_t vftr_clock = CLOCK_MONOTONIC;

// set the local reference time to which all 
// timedifferences are measured
struct timespec local_ref_time = {0ll,0ll};
void vftr_set_local_ref_time() {
   // simply get the current time and store it
   clock_gettime(vftr_clock, &local_ref_time);
   return;
}

// return the local reverence time
long long vftr_get_local_ref_time() {
   return 0;
}

// get the current time in micro seconds since
// the reference time point
long long vftr_get_runtime_usec() {
   // get the current time
   struct timespec local_time;
   clock_gettime(vftr_clock, &local_time);

   // compute the time difference in microseconds
   // difference in second counter
   long long delta_sec  = local_time.tv_sec  - local_ref_time.tv_sec;
   // difference in nanosecond counter
   long long delta_nsec = local_time.tv_nsec - local_ref_time.tv_nsec;
   // handle a possible carry over of nanoseconds
   if (delta_nsec < 0) {
      // add one second in nano seconds to the nano second difference
      delta_nsec += 1000000000l;
      // substract one second in microseconds from microsecond difference
      delta_sec -= 1l;
   }

   // return the amount of microseconds since the local reference time
   return 1000000l*delta_sec + delta_nsec/1000l;
}

// get the number of elapsed clock counts
unsigned long long vftr_get_cycles () {
  unsigned long long ret;
#if defined (__ve__)
// On SX Aurora, we obtain the number of elapsed cycles by reading the usrcc register
// using the smir command. In earlier versions, lhm.l was used. However, this yields
// VH clock counts, which is not the correct measure to compute e.g. the vector time.
  asm volatile ("smir %0, %usrcc" : "=r"(ret));
#elif defined (__x86_64__)
    unsigned int a, d;
    asm volatile("rdtsc" : "=a" (a), "=d" (d));
    ret = ((unsigned long long)a) | (((unsigned long long)d) << 32);
#endif
  return ret;
}
