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

#ifndef VFTR_OPENMP_H
#define VFTR_OPENMP_H

#ifdef _OPENMP

#include <omp.h>
#define OMP_GET_THREAD_NUM  omp_get_thread_num()
#define OMP_GET_MAX_THREADS omp_get_max_threads()
#else
typedef int omp_lock_t;
#define OMP_GET_THREAD_NUM  0
#define OMP_GET_MAX_THREADS 1
#endif

omp_lock_t vftr_lock;                 /* Lock for debug print separation */
omp_lock_t vftr_lock_exp;             /* Lock for experiments */
omp_lock_t vftr_lock_hook;            /* Lock for DrHook */
omp_lock_t vftr_lock_prof;            /* Lock for profile print separation */


#endif
