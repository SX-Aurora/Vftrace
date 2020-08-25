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

#ifndef VFTR_ENVIRONMENT_H
#define VFTR_ENVIRONMENT_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "vftr_regex.h"

#define VFTR_ENV_VAR_MAX_LENGTH 50

typedef struct env_var_int {
	int value;
	bool set;	
} env_var_int_t;

typedef struct env_var_long {
	long value;
	bool set;	
} env_var_long_t;

typedef struct env_var_long_long {
	long long value;
	bool set;	
} env_var_long_long_t;

typedef struct env_var_double {
	double value;
	bool set;	
} env_var_double_t;

typedef struct env_var_bool {
	bool value;
	bool set;	
} env_var_bool_t;

typedef struct env_var_string {
	char *value;
	bool set;	
} env_var_string_t;

typedef struct env_var_regex {
	regex_t *value;
	bool set;	
} env_var_regex_t;

typedef struct vftr_envs {
	env_var_bool_t *vftrace_off;
	env_var_bool_t *do_sampling;
	env_var_bool_t *regions_precise;
        env_var_string_t *output_directory;
	env_var_string_t *logfile_basename;
        env_var_bool_t *logfile_all_ranks;
// default sample time in s as a floating point number
	env_var_double_t *sampletime;
// maximum runtime in seconds (default: a week)
	env_var_long_long_t *stoptime;
	env_var_bool_t *accurate_profile;	
	env_var_bool_t *prof_truncate;
	env_var_bool_t *mpi_log;	
	env_var_bool_t *signals_off;
	env_var_int_t *bufsize;
	env_var_regex_t *runtime_profile_funcs;
	env_var_regex_t *include_only_regex;
	env_var_regex_t *exclude_functions_regex;
	env_var_double_t *detail_until_cum_cycles;
	env_var_string_t *scenario_file;
	env_var_regex_t *preciseregex;
	env_var_bool_t *license_verbose;
        env_var_string_t *print_stacks_for;
        env_var_string_t *print_loadinfo_for;
} vftr_envs_t;

extern vftr_envs_t *vftr_environment;

void vftr_read_environment();
void vftr_assert_environment();
void vftr_free_environment();
bool vftr_off();
void vftr_switch_off();
bool vftr_env_do_sampling();

int vftr_environment_test_1 (FILE *fp);

#endif
