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

#define SCENARIO_NAME_LEN 32

typedef struct EventCounter {
    int                 rank, id, namelen, decipl;
    long long           *count;
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

void vector_engine_compute_counters (long long *regs, double current_runtime,
				double *mflops, double *vlen, double *pvec, double *pbank);

void vftr_write_scenario_header_to_vfd (FILE *fp);
void vftr_write_observables_to_vfd (long long cycles, FILE *fp);

#define TE_MAX 50
#define SCENARIO_EXPR_BUF_SIZE 200

typedef struct {
	char *unit;
	char *group;
	char *column1;
	char *column2;
	int decpl_1;
	int decpl_2;
} hwc_format_t;


typedef struct {
	char *name;
	char *formula;
	char *protected_values;
	double default_value;
	double value;
	int integrated;
} function_expr_t;
	
char *scenario_expr_counter_names[TE_MAX];
char *scenario_expr_vars[TE_MAX];
function_expr_t scenario_expr_formulas[TE_MAX];
hwc_format_t scenario_expr_format[TE_MAX];

int scenario_expr_n_vars;
int scenario_expr_n_formulas;

double *scenario_expr_counter_values;
double scenario_expr_runtime;
double scenario_expr_cycles;
double scenario_expr_cycletime;

int vftr_read_scenario_file (char *filename);
void scenario_expr_evaluate (int i_scenario, double runtime, double cycles);
void scenario_expr_evaluate_all (double runtime, double cycles);
void scenario_expr_print_summary (FILE *fp);
void scenario_expr_print_raw_counters (FILE *fp);
double scenario_expr_get_value (int i_scenario);
void scenario_expr_set_formats ();
void scenario_expr_print_column (FILE *f, int i_scenario);
void scenario_expr_print_all_columns (FILE *f);
int scenario_expr_get_table_width ();
int scenario_expr_get_column_width (int i_scenario);
void scenario_expr_unique_group_indices (int *n_groups, int *is_unique, int id);
void scenario_expr_print_header (FILE *fp);
void scenario_expr_print_group (FILE *fp);
void scenario_expr_print_subgroup (FILE *fp);
void scenario_expr_add_papi_counters ();
void scenario_expr_add_veperf_counters ();

#endif
