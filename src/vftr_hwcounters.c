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

#if defined(HAS_PAPI)
#include "papi.h"
#endif
#include "vftr_filewrite.h"

#include "vftr_environment.h"
#include "vftr_hwcounters.h"
#include "vftr_scenarios.h"
#include "vftr_stacks.h"
#include "vftr_setup.h"
#include "vftr_mallinfo.h"

int vftr_find_event_number (char *);

evtcounter_t *first_counter = NULL;
evtcounter_t *next_counter = NULL;

int vftr_n_hw_obs;
bool vftr_events_enabled;
long long vftr_prog_cycles;
bool err_no_hwc_support = false;
#ifdef HAS_PAPI
int vftr_papi_event_set;
#endif

long long vftr_echwc[MAX_HWC_EVENTS];

// As each hardware observable is registered, this counter is incremented. 

void vftr_new_counter (char *name, int id, int rank) {
    evtcounter_t *evc;
    evc = (evtcounter_t *) malloc (sizeof(evtcounter_t));
    evc->name = strdup (name);
    evc->namelen = strlen (name);
    evc->count = 0ll;
    evc->next = NULL;
    evc->decipl  = 1;
    evc->id = id;

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
    vftr_new_counter (name, id, 0);
}

/**********************************************************************/

void vftr_sx_counter (char *name, int id) {
    vftr_new_counter (name, id, 0);
}

/**********************************************************************/

void vftr_internal_counter (char *name, int id) {
   vftr_new_counter (name, id, 0);
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
    vftr_papi_event_set = PAPI_NULL;
    char errmsg[256];
    if ((diag = PAPI_create_eventset(&vftr_papi_event_set)) != PAPI_OK) {
       PAPI_perror (errmsg);
       fprintf(vftr_log, "vftr_init_hwcounters - "
               "PAPI_create_eventset error: %s\n", errmsg);
    }
}
#endif

/**********************************************************************/

#ifdef HAS_PAPI
int eventset_is_filled () {
	return vftr_papi_event_set != PAPI_NULL;
}
#endif

/**********************************************************************/

#ifdef HAS_PAPI
void vftr_start_hwcounters () {
    evtcounter_t *evc;

    if (err_no_hwc_support) return;
    vftr_init_papi_counters();

    char errmsg[256];
    evtcounter_t *e;
    int diag;

    e = first_counter;
    for (int i = 0; e; i++) {
           if ((diag = PAPI_add_event(vftr_papi_event_set, e->id)) != PAPI_OK) {
	       PAPI_perror (errmsg);
	       fprintf(vftr_log, "vftr_start_hwcounters - "
                                 "PAPI_add_event error: %s when adding %s\n",
                                 errmsg, e->name );
           }
           e = e->next;
    }
    if (eventset_is_filled()) {
    	if ((diag = PAPI_start(vftr_papi_event_set)) != PAPI_OK) {
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

    vftr_n_hw_obs = 0;
    vftr_init_scenario_formats ();
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

    for (int i = 0; i < vftr_scenario_expr_n_vars; i++) {
#if defined(HAS_SXHWC)
       vftr_sx_counter (vftr_scenario_expr_counter_names[i], i);
#elif defined(HAS_PAPI)
       vftr_papi_counter (vftr_scenario_expr_counter_names[i]);
#endif
    }

    if (vftr_memtrace) {
       vftr_internal_counter ("Memory", vftr_scenario_expr_n_vars);
    }

#if defined(HAS_PAPI)
    vftr_start_hwcounters();
#endif
    return 0;
}

/**********************************************************************/

#if defined(HAS_SXHWC)
void vftr_read_sxhwc_registers (long long hwc[MAX_HWC_EVENTS]) {
    long long tmp[16];
    //printf ("READ REGISTERS\n");
    asm volatile (
        "smir %0,  %%pmc0\n\t"
        "smir %1,  %%pmc1\n\t"
        "smir %2,  %%pmc2\n\t"
        "smir %3,  %%pmc3\n\t"
        "smir %4,  %%pmc4\n\t"
        "smir %5,  %%pmc5\n\t"
        "smir %6,  %%pmc6\n\t"
        "smir %7,  %%pmc7\n\t"
        "smir %8,  %%pmc8\n\t"
        "smir %9,  %%pmc9\n\t"
        "smir %10, %%pmc10\n\t"
        "smir %11, %%pmc11\n\t"
        "smir %12, %%pmc12\n\t"
        "smir %13, %%pmc13\n\t"
        "smir %14, %%pmc14\n\t"
        "smir %15, %%pmc15\n\t"
        :
        "=r"(hwc[0]),
        "=r"(hwc[1]),
        "=r"(hwc[2]),
        "=r"(hwc[3]),
        "=r"(hwc[4]),
        "=r"(hwc[5]),
        "=r"(hwc[6]),
        "=r"(hwc[7]),
        "=r"(hwc[8]),
        "=r"(hwc[9]),
        "=r"(hwc[10]),
        "=r"(hwc[11]),
        "=r"(hwc[12]),
        "=r"(hwc[13]),
        "=r"(hwc[14]),
        "=r"(hwc[15])
    );
}

/**********************************************************************/

void vftr_read_counters_sx (long long *event) {
    int i, j, diag;
    evtcounter_t *evc;
    if (event == NULL) return;
    vftr_read_sxhwc_registers (vftr_echwc);
    memset (vftr_scenario_expr_counter_values, 0., sizeof(double) * vftr_scenario_expr_n_vars);
    /* Mask overflow bit and undefined bits */
    vftr_echwc[0] &= 0x000fffffffffffff; /* 52bit counter */
    vftr_echwc[1] &= 0x000fffffffffffff; /* 52bit counter */
    for (i = 2; i < 16; i++) vftr_echwc[i] &= 0x00ffffffffffffff; /* 56bit counter */
    for (int i = 0; i < vftr_scenario_expr_n_vars; i++) {
	vftr_scenario_expr_counter_values[i] = vftr_echwc[i];
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
    if (vftr_scenario_expr_n_vars > 0) {
        if (vftr_papi_event_set != PAPI_NULL) {
            if ((diag = PAPI_read(vftr_papi_event_set, vftr_echwc)) != PAPI_OK) {
                fprintf(vftr_log, "error: PAPI_read returned %d\n", diag);
    	}
        }
        for (j = 0, evc = first_counter; evc; evc = evc->next) {
            evc->count = vftr_echwc[j++];
        }
    }
    for (i = 0, evc = first_counter; evc; i++, evc = evc->next) {
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
    long long ec[MAX_HWC_EVENTS];
    if ((diag = PAPI_stop(vftr_papi_event_set, ec)) != PAPI_OK)
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

int vftr_sxhwc_test_1 (FILE *fp_in, FILE *fp_out) {
#if defined(HAS_SXHWC)
#define N_DIGITS 6
	const char *sx_counter_names[16] = {"EX", "VX", "FPEC", "VE", "VECC", "L1MCC", 
		"VE2", "VAREC", "VLDEC", "PCCC", "VLPC", "VLEC", "VLCME", "FMAEC", "PTCC", "TTCC"};
	int n = 100000;
	int n_iter = 1000;
	double x[n], y[n], z[n];
	long long *c1, *c2;
	long long c_diff[MAX_HWC_EVENTS][n_iter];
	c1 = (long long *)malloc (16 * sizeof(long long));
	c2 = (long long *)malloc (16 * sizeof(long long));
	fprintf (fp_out, "Checking reproducibility of SX Aurora hardware counters\n");
	fprintf (fp_out, "Averaging over %d iterations\n", n_iter);

	for (int i = 0; i < n; i++) {
		x[i] = i;
		y[i] = 0.5 * i;
	}
	for (int n = 0; n < n_iter; n++) {
		vftr_read_sxhwc_registers (c1);
		for (int i = 0; i < n; i++) {
			z[i] = x[i] + x[i] * y[i];
		}
		vftr_read_sxhwc_registers (c2);
		for (int i = 0; i < 16; i++) {
			c_diff[i][n] = c2[i] - c1[i];
		}
	}
	
	double c_avg[MAX_HWC_EVENTS];
	long long sum_c;
	for (int i = 0; i < 16; i++) {
		sum_c = 0;
		for (int n = 0; n < n_iter; n++) {
			sum_c += c_diff[i][n];
		}
		c_avg[i] = (double)sum_c / n_iter;
	}
        // The counters ending with a "C" are clock counters. They depend on the momentary performance of
        // the system. Therefore, the mean value is not reliable and is therefore not printed. We constrain
        // this output only to the hardware counters without a "C".
	fprintf (fp_out, "%*s: %*d\n", N_DIGITS, sx_counter_names[0], N_DIGITS, (int)floor(c_avg[0])); // EX
	fprintf (fp_out, "%*s: %*d\n", N_DIGITS, sx_counter_names[1], N_DIGITS, (int)floor(c_avg[1])); // VX 
	fprintf (fp_out, "%*s: %*d\n", N_DIGITS, sx_counter_names[3], N_DIGITS, (int)floor(c_avg[3])); // VE
	fprintf (fp_out, "%*s: %*d\n", N_DIGITS, sx_counter_names[6], N_DIGITS, (int)floor(c_avg[6])); // VE2
	fprintf (fp_out, "%*s: %*d\n", N_DIGITS, sx_counter_names[12], N_DIGITS, (int)floor(c_avg[12])); // VLCME
	free(c1);
	free(c2);
#endif
	return 0;
}

/**********************************************************************/
