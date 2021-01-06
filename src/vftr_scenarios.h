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

#ifndef SCENARIOS_H
#define SCENARIOS_H

#include <stdio.h>
#include <stdbool.h>
#include "vftr_functions.h"

#define SCENARIO_NAME_LEN 32

typedef struct EventCounter {
    int                 rank, id, namelen, decipl;
    long long           count;
    char                *name, fmt[10];
    struct EventCounter *next;
} evtcounter_t;

evtcounter_t  *vftr_get_counters ( void );

enum ve_counters {
	EX, // 0
	VX,
	FPEC, // 2
	VE,
	VECC, // 4
	L1MCC,
	VE2, // 6
	VAREC,
	VLDEC, // 8
	PCCC,
	VLDCC, // 10
	VLEC,
	VLECME, // 12
	FMAEC,
	PTCC, // 14
	TTCC
};	

void vftr_write_scenario_header_to_vfd (FILE *fp);
void vftr_write_observables_to_vfd (profdata_t *prof_current, profdata_t *prof_previous, FILE *fp);

#define TE_MAX 50
#define SCENARIO_EXPR_BUF_SIZE 200

typedef struct {
	char *header;
	char *unit;
	int decimal_places;
} hwc_format_t;


typedef struct {
	char *name;
	char *formula;
	char *protected_values;
	double default_value;
	double value;
	bool integrated;
} function_expr_t;
	
extern char *vftr_scenario_expr_counter_names[TE_MAX];
extern char *vftr_scenario_expr_vars[TE_MAX];
extern function_expr_t vftr_scenario_expr_formulas[TE_MAX];
extern hwc_format_t vftr_scenario_expr_format[TE_MAX];

extern int vftr_scenario_expr_n_vars;
extern int vftr_scenario_expr_n_formulas;

extern double *vftr_scenario_expr_counter_values;
extern double vftr_scenario_expr_runtime;
extern double vftr_scenario_expr_cycles;
extern double vftr_scenario_expr_cycletime;

void vftr_init_scenario_formats ();
int vftr_read_scenario_file (char *filename, FILE *fp_ext);
void vftr_scenario_expr_evaluate (int i_scenario, double runtime, unsigned long long cycles);
void vftr_scenario_expr_evaluate_all (double runtime, unsigned long long cycles);
void vftr_scenario_expr_print_summary (FILE *fp);
void vftr_scenario_expr_print_raw_counters (FILE *fp);
double vftr_scenario_expr_get_value (int i_scenario);
void vftr_scenario_expr_set_formats ();
void vftr_scenario_expr_print_column (FILE *f, int i_scenario);
void vftr_scenario_expr_print_all_columns (FILE *f);
int vftr_scenario_expr_get_table_width ();
int vftr_scenario_expr_get_column_width (int i_scenario);
void vftr_scenario_expr_unique_group_indices (int *n_groups, int *is_unique, int id);
void vftr_scenario_expr_print_header (FILE *fp);
void vftr_scenario_expr_print_group (FILE *fp);
void vftr_scenario_expr_print_subgroup (FILE *fp);
void vftr_scenario_expr_add_papi_counters ();
void vftr_scenario_expr_add_sx_counters ();

// test functions
int vftr_scenario_test_1 (FILE *fp_in, FILE *fp_out);
int vftr_scenario_test_2 (FILE *fp_in, FILE *fp_out);
int vftr_scenario_test_3 (FILE *fp_in, FILE *fp_out);

#endif
