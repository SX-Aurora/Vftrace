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

#ifndef VFTR_MONITORING_THREAD_H
#define VFTR_MONITORING_THREAD_H

#include <stdbool.h>
#include <pthread.h>

// Monitoring values
typedef struct {
   long max_memory;
} vftr_monitored_values_t;

// lock for thread synchronisation
extern pthread_mutex_t vftr_lock_monitor_thread;

void vftr_create_monitor_thread(long long monitor_interval_usec);

void vftr_join_monitor_thread();

// Use these functions to lock the monitoring thread for access on shared data
void vftr_lock_monitoring_thread();
void vftr_unlock_monitor_thread();

//
vftr_monitored_values_t vftr_get_and_reset_monitored_values();

#endif
