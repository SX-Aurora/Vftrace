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
#include <limits.h>
#include <stdio.h>
#include <string.h>

#if defined(HAS_VEPERF)
#include <stdint.h>
#include <veperf.h>
#include "vftr_filewrite.h"
#elif defined(HAS_PAPI)
#include "papi.h"
#include "vftr_filewrite.h"
#endif

#include "vftr_omp.h"
#include "vftr_environment.h"
#include "vftr_hwcounters.h"
#include "vftr_scenarios.h"
#include "vftr_stacks.h"

int vftr_find_event_number (char *);

evtcounter_t *first_counter = NULL;
evtcounter_t *next_counter = NULL;

int vftr_n_hw_obs;
bool vftr_events_enabled;
long long *vftr_prog_cycles;
bool err_no_hwc_support = false;
int  *eventSet;

long long vftr_echwc[MAX_HWC_EVENTS];

// As each hardware observable is registered, this counter is incremented. 
int hwc_event_num = 0;

// Should be allocated to hwc_event_num ?
long long vftr_echwc[MAX_HWC_EVENTS];

void vftr_new_counter (char *name, int id, int rank) {
    evtcounter_t *evc;
    evc = (evtcounter_t *) malloc( sizeof( evtcounter_t ) );
    evc->name    = strdup( name );
    evc->namelen = strlen( name );
    evc->count   = (long long *) malloc( vftr_omp_threads * sizeof( long long ) );
    evc->next    = NULL;
    evc->decipl  = 1;
    evc->id      = id;
    evc->rank    = rank;

    if (!first_counter) {
	first_counter = next_counter = evc;
    } else {
	next_counter = next_counter->next = evc;
    }

    memset (evc->count, 0, vftr_omp_threads * sizeof(long long));
    vftr_n_hw_obs++;
}

/**********************************************************************/

void vftr_papi_counter (char *name) {
    int id   = err_no_hwc_support ? -1 : vftr_find_event_number (name);
    vftr_new_counter (name, id, hwc_event_num++);
}

/**********************************************************************/

void vftr_sx_counter (char *name, int id) {
    vftr_new_counter( name, id, hwc_event_num++);
}

/**********************************************************************/

#if defined(HAS_PAPI)
void vftr_init_papi_counters () {
    int i, diag;

    if((diag = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
        fprintf(vftr_log, "vftr_init_counters error: "
                          "PAPI_library_init returned %d, "
                          "no h/w event counter support\n", diag);
        err_no_hwc_support = true;
        return;
    }
    eventSet = (int *) malloc (vftr_omp_threads * sizeof(int));
    for (i = 0; i < vftr_omp_threads; i++) {
    	eventSet[i] = PAPI_NULL;
    }
#ifdef _OPENMP
    if ((diag = PAPI_thread_init(
                 (unsigned long (*)(void))omp_get_thread_num)) != PAPI_OK) {
        fprintf(vftr_log, "vftr_init_hwcounters - "
                          "PAPI_thread_init error code: %d\n", diag);
        return;
    }
#pragma omp parallel
#endif
    {
        char errmsg[256];
        int diag, me = OMP_GET_THREAD_NUM;
        if ((diag = PAPI_create_eventset(&eventSet[me])) != PAPI_OK) {
    		PAPI_perror( errmsg );
            	fprintf(vftr_log, "vftr_init_hwcounters - "
                              "PAPI_create_eventset error: %s\n", errmsg);
        }
    }
}
#endif

/**********************************************************************/

#ifdef HAS_PAPI
int eventset_is_filled () {
	int i = 0;
	int is_filled = 0;
	while (i < vftr_omp_threads && !is_filled) {
		is_filled = eventSet[i++] != PAPI_NULL;
	}
	return is_filled;
}
#endif

/**********************************************************************/

#ifdef HAS_PAPI
void vftr_start_hwcounters () {
    evtcounter_t *evc;

    if (err_no_hwc_support) return;

#ifdef _OPENMP
#pragma omp parallel
#endif
{
    char errmsg[256];
    evtcounter_t *e;
    int diag, omp_thread = OMP_GET_THREAD_NUM;
    int i;

    for (i = 0, e = first_counter; e; i++, e = e->next) {
           if ((diag = PAPI_add_event(eventSet[omp_thread], e->id)) != PAPI_OK) {
	       PAPI_perror( errmsg );
	       fprintf(vftr_log, "vftr_start_hwcounters - "
                                 "PAPI_add_event error: %s when adding %s\n",
                                 errmsg, e->name );
           }
    }
    if (eventset_is_filled()) {
    	if ((diag = PAPI_start(eventSet[omp_thread])) != PAPI_OK) {
    	    PAPI_perror( errmsg );
    	    fprintf(stdout, "vftr_start_hwcounters - PAPI_start error: %s\n", errmsg);
    	}
    }
}
}
#endif

/**********************************************************************/

int vftr_init_hwc (char *scenario_file) {
    int i;
    char *c, *s;
    evtcounter_t *evc;
#if defined(HAS_PAPI)
    const PAPI_hw_info_t        *hwinfo;
#endif

#if defined(HAS_VEPERF)
    int stat = __veperf_init();
    if (stat) fprintf( vftr_log, "vftr_init_hwc: __veperf_init() failed (%d)\n", stat );
#endif

    vftr_n_hw_obs = 0;
    if (vftr_read_scenario_file (scenario_file)) {
	return -1;
    }

#if defined(HAS_PAPI)
    vftr_start_hwcounters();
    hwinfo  = PAPI_get_hardware_info();
    if (hwinfo == NULL) {
        err_no_hwc_support = true;
        return -2;
    }
    int vftr_cpumodel   = hwinfo->model;
    // TODO: Compare with cpu given in the model file
#endif

#if defined(HAS_VEPERF)
    scenario_expr_add_veperf_counters ();
#elif defined(HAS_PAPI)
    scenario_expr_add_papi_counters ();
#endif

#if defined(HAS_PAPI)
    vftr_start_hwcounters();
#endif
    return 0;
}

/**********************************************************************/

#if defined(HAS_VEPERF)
void vftr_read_counters_veperf (long long *event, int omp_thread) {
    int i, j, diag;
    evtcounter_t *evc;
    if (event == NULL) return;
    veperf_get_pmcs ((int64_t *)vftr_echwc);
    memset (scenario_expr_counter_values, 0., sizeof(double) * scenario_expr_n_vars);
    /* Mask overflow bit and undefined bits */
    vftr_echwc[0] &= 0x000fffffffffffff; /* 52bit counter */
    vftr_echwc[1] &= 0x000fffffffffffff; /* 52bit counter */
    for (i = 2; i < 16; i++) vftr_echwc[i] &= 0x00ffffffffffffff; /* 56bit counter */
    for (int i = 0; i < scenario_expr_n_vars; i++) {
	scenario_expr_counter_values[i] = vftr_echwc[i];
    }
    for (i = j  = 0, evc = first_counter; evc; i++, evc = evc->next) {
        evc->count[omp_thread] = vftr_echwc[j++];
        event[i] = evc->count[omp_thread];
    }

}
#endif

/**********************************************************************/

#if defined(HAS_PAPI)
void vftr_read_counters_papi (long long *event, int omp_thread) {
    int i, j, diag;
    evtcounter_t *evc;
    if (event == NULL) return;
    if (hwc_event_num > 0) {
        if (eventSet[omp_thread] != PAPI_NULL) {
            if ((diag = PAPI_read(eventSet[omp_thread], vftr_echwc)) != PAPI_OK) {
                fprintf(vftr_log, "[%d] error: PAPI_read returned %d\n", omp_thread, diag);
    	}
        }
        for (j = 0,evc = first_counter; evc; evc = evc->next) {
            evc->count[omp_thread] = vftr_echwc[j++];
        }
    }
    for (i = 0,evc = first_counter; evc; i++, evc = evc->next) {
        event[i] = evc->count[omp_thread];
    }
}
#endif

/**********************************************************************/

// We need a dummy symbol if neither PAPI nor VEPERF is used
void vftr_read_counters_dummy (long long *event, int omp_thread) {
}

/**********************************************************************/

int vftr_stop_hwc () {
    int diag = 0;
#if defined(HAS_PAPI)
#ifdef _OPENMP_OFF
#pragma omp parallel
#endif
{
    int diag, me = OMP_GET_THREAD_NUM;
    long long ec[MAX_HWC_EVENTS];
    if (( diag = PAPI_stop(eventSet[me], ec)) != PAPI_OK)
    fprintf(vftr_log, "vftr_stop_hwc error: PAPI_stop returned %d\n", diag);
}
#endif
    return diag;
}

/**********************************************************************/

evtcounter_t  *vftr_get_counters( void ) {
    return first_counter;
}

/**********************************************************************/

int vftr_find_event_number (char *s) {
#if defined(HAS_PAPI)
    int i, stat;
    if ((stat = PAPI_event_name_to_code(strdup(s), &i)) == PAPI_OK) {
       return i;
    }
#endif
    return -1;
}

unsigned long long vftr_get_cycles () {
  unsigned long long hrTime;
#if defined (__ve__)
  void *vehva = (void *)0x000000001000;
  asm volatile ("lhm.l %0,0(%1)" : "=r" (hrTime) : "r" (vehva));
#endif
  return hrTime;
}
