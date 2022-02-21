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

#ifndef OMP_TIMER_H
#define OMP_TIMER_H

#include <stdlib.h>

#include <vftr_timer.h>

extern long long vftr_omp_overhead_usec;

extern int vftr_omp_ntimer;
extern long long *vftr_omp_time_usec;
extern long long *vftr_omp_wait_time_usec;

void vftr_omp_timer_realloc(int ithread);

void vftr_omp_time_add(int ithread, long long timeslice);

void vftr_omp_wait_time_add(int ithread, long long timeslice);

#endif
