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

#include <vftr_timer.h>

long long vftr_omp_overhead_usec = 0ll;

int vftr_omp_ntimer = 0;
long long *vftr_omp_time_usec = NULL;
long long *vftr_omp_wait_time_usec = NULL;

void vftr_omp_timer_realloc(int ithread) {
   if (ithread >= vftr_omp_ntimer) {
      int ntimer = ithread+1;
      vftr_omp_time_usec =
         (long long*) realloc(vftr_omp_time_usec, ntimer*sizeof(long long));
      vftr_omp_wait_time_usec =
         (long long*) realloc(vftr_omp_wait_time_usec, ntimer*sizeof(long long));
      for (int jthread=vftr_omp_ntimer; jthread<ntimer; jthread++) {
         vftr_omp_time_usec[jthread] = 0ll;
         vftr_omp_wait_time_usec[jthread] = 0ll;
      }
      vftr_omp_ntimer = ntimer;
   }
}

void vftr_omp_time_add(int ithread, long long timeslice) {
   vftr_omp_timer_realloc(ithread);
   vftr_omp_time_usec[ithread] += timeslice;
}

void vftr_omp_wait_time_add(int ithread, long long timeslice) {
   vftr_omp_timer_realloc(ithread);
   vftr_omp_wait_time_usec[ithread] += timeslice;
}
