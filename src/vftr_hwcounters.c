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
#include <math.h>

#if defined(HAS_VEPERF)
#include <stdint.h>
#include <veperf.h>
#include "vftr_filewrite.h"
#elif defined(HAS_PAPI)
#include "papi.h"
#include "vftr_filewrite.h"
#endif

#include "vftr_environment.h"
#include "vftr_hwcounters.h"
#include "vftr_scenarios.h"
#include "vftr_stacks.h"

int vftr_find_event_number (char *);

evtcounter_t *first_counter = NULL;
evtcounter_t *next_counter = NULL;

int vftr_n_hw_obs;
bool vftr_events_enabled;
long long vftr_prog_cycles;
bool err_no_hwc_support = false;
#ifdef HAS_PAPI
int papi_event_set;
#endif

long long vftr_echwc[MAX_HWC_EVENTS];

// As each hardware observable is registered, this counter is incremented. 
int hwc_event_num = 0;

// Should be allocated to hwc_event_num ?
long long vftr_echwc[MAX_HWC_EVENTS];

void vftr_new_counter (char *name, int id, int rank) {
    evtcounter_t *evc;
    evc = (evtcounter_t *) malloc (sizeof(evtcounter_t));
    evc->name = strdup (name);
    evc->namelen = strlen (name);
    evc->count = 0ll;
    evc->next = NULL;
    evc->decipl  = 1;
    evc->id = id;
    evc->rank = rank;

    if (!first_counter) {
	first_counter = next_counter = evc;
    } else {
	next_counter = next_counter->next = evc;
    }

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
    papi_event_set = PAPI_NULL;
    char errmsg[256];
    if ((diag = PAPI_create_eventset(papi_event_set)) != PAPI_OK) {
       PAPI_perror (errmsg);
       fprintf(vftr_log, "vftr_init_hwcounters - "
               "PAPI_create_eventset error: %s\n", errmsg);
    }
}
#endif

/**********************************************************************/

#ifdef HAS_PAPI
int eventset_is_filled () {
	return papi_event_set != PAPI_NULL;
}
#endif

/**********************************************************************/

#ifdef HAS_PAPI
void vftr_start_hwcounters () {
    evtcounter_t *evc;

    if (err_no_hwc_support) return;

    char errmsg[256];
    evtcounter_t *e;
    int diag;

    for (int i = 0, e = first_counter; e; i++, e = e->next) {
           if ((diag = PAPI_add_event(papi_event_set, e->id)) != PAPI_OK) {
	       PAPI_perror (errmsg);
	       fprintf(vftr_log, "vftr_start_hwcounters - "
                                 "PAPI_add_event error: %s when adding %s\n",
                                 errmsg, e->name );
           }
    }
    if (eventset_is_filled()) {
    	if ((diag = PAPI_start(papi_event_set)) != PAPI_OK) {
    	    PAPI_perror( errmsg );
    	    fprintf(stdout, "vftr_start_hwcounters - PAPI_start error: %s\n", errmsg);
    	}
    }
}
#endif

/**********************************************************************/

int vftr_init_hwc (char *scenario_file) {
#if defined(HAS_PAPI)
    const PAPI_hw_info_t        *hwinfo;
#endif

#if defined(HAS_VEPERF)
    int stat = __veperf_init();
    if (stat) fprintf( vftr_log, "vftr_init_hwc: __veperf_init() failed (%d)\n", stat );
#endif

    vftr_n_hw_obs = 0;
    if (vftr_read_scenario_file (scenario_file, NULL)) {
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
void vftr_read_counters_veperf (long long *event) {
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
        evc->count = vftr_echwc[j++];
        event[i] = evc->count;
    }

}
#endif

/**********************************************************************/

#if defined(HAS_PAPI)
void vftr_read_counters_papi (long long *event) {
    int i, j, diag;
    evtcounter_t *evc;
    if (event == NULL) return;
    if (hwc_event_num > 0) {
        if (papi_event_set != PAPI_NULL) {
            if ((diag = PAPI_read(vftr_echwc)) != PAPI_OK) {
                fprintf(vftr_log, "error: PAPI_read returned %d\n", diag);
    	}
        }
        for (j = 0,evc = first_counter; evc; evc = evc->next) {
            evc->count = vftr_echwc[j++];
        }
    }
    for (i = 0,evc = first_counter; evc; i++, evc = evc->next) {
        event[i] = evc->count;
    }
}
#endif

/**********************************************************************/

// We need a dummy symbol if neither PAPI nor VEPERF is used
void vftr_read_counters_dummy (long long *event) {
}

/**********************************************************************/

int vftr_stop_hwc () {
    int diag = 0;
#if defined(HAS_PAPI)
    int diag;
    long long ec[MAX_HWC_EVENTS];
    if ((diag = PAPI_stop(papi_event_set, ec)) != PAPI_OK)
    fprintf(vftr_log, "vftr_stop_hwc error: PAPI_stop returned %d\n", diag);
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

/**********************************************************************/
int vftr_veperf_test_1 (FILE *fp_in, FILE *fp_out) {
#if defined(HAS_VEPERF)
	//int n = 1000;
	//double x[n], y[n], z[n];
	int stat = __veperf_init();
	if (stat) {
		fprintf (fp_out, "__veperf_init() failed(%d)\n", stat);	  
		return stat;
	}
	veperf_get_pmcs ((int64_t *)vftr_echwc);
	fprintf (fp_out, "veperf: success\n");
#endif
	return 0;
}

/**********************************************************************/

int vftr_veperf_test_2 (FILE *fp_in, FILE *fp_out) {
#if defined(HAS_VEPERF)
	int n = 1000;
	int n_iter = 50;
	double x[n], y[n], z[n];
	long long c1[MAX_HWC_EVENTS], c2[MAX_HWC_EVENTS], c_diff[MAX_HWC_EVENTS][n_iter];
	fprintf (fp_out, "Checking reproducibility of SX Aurora hardware counters\n");
	fprintf (fp_out, "Averaging over %d iterations\n", n_iter);
	for (int i = 0; i < n; i++) {
		x[i] = i;
		y[i] = 0.5 * i;
	}
	int stat = __veperf_init();
	if (stat) {
		fprintf (fp_out, "__veperf_init() failed(%d)\n", stat);	  
		return stat;
	}
	for (int n = 0; n < n_iter; n++) {
		veperf_get_pmcs ((int64_t *)c1);
		for (int i = 0; i < n; i++) {
			z[i] = x[i] + x[i] * y[i];
		}
		veperf_get_pmcs ((int64_t *)c2);
		for (int i = 0; i < 15; i++) {
			c_diff[i][n] = c2[i] - c1[i];
		}
	}
	
	double c_avg[MAX_HWC_EVENTS];
	long long sum_c;
	for (int i = 0; i < 15; i++) {
		sum_c = 0;
		for (int n = 0; n < n_iter; n++) {
			sum_c += c_diff[i][n];
		}
		c_avg[i] = (double)sum_c / n_iter;
	}
	for (int i = 0; i < 15; i++) {
		fprintf (fp_out, "i: %d, c: %d\n", i, (int)floor(c_avg[i]));
	}
#endif
	return 0;
}

/**********************************************************************/
