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

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <ctype.h>

#include "vftr_hwcounters.h"
#include "vftr_environment.h"

vftr_envs_t *vftr_environment;

env_var_int_t *vftr_read_env_int (char *env_name, int val_default) {
    char *s;
    env_var_int_t *var;
    var = (env_var_int_t*)malloc (sizeof (env_var_int_t));
    if (s = getenv (env_name)) {
	var->value = atoi(s);
 	var->set = true;
    } else {
	var->value = val_default;
        var->set = false;
    }
    return var;
}

env_var_long_t *vftr_read_env_long (char *env_name, long val_default) {
    char *s;
    env_var_long_t *var;
    var = (env_var_long_t*)malloc (sizeof (env_var_long_t));
    if (s = getenv (env_name)) {
	var->value = atol(s);
 	var->set = true;
    } else {
	var->value = val_default;
        var->set = false;
    }
    return var;
}

env_var_long_long_t *vftr_read_env_long_long (char *env_name, long long val_default) {
    char *s;
    env_var_long_long_t *var;
    var = (env_var_long_long_t*)malloc (sizeof (env_var_long_long_t));
    if (s = getenv (env_name)) {
	var->value = atoll(s);
 	var->set = true;
    } else {
	var->value = val_default;
        var->set = false;
    }
    return var;
}

env_var_bool_t *vftr_read_env_bool (char *env_name, bool val_default) {
    env_var_bool_t *var;
    var = (env_var_bool_t*)malloc (sizeof (env_var_bool_t));
    char *s;
    if (s = getenv (env_name)) {
        // convert string to only lowercase to ease comparison
        char *tmps = s;
        while (*tmps) {
           *tmps = tolower(*tmps);
           tmps++;
        }
        if (!strcmp(s, "1") ||
            !strcmp(s, "yes") ||
            !strcmp(s, "on")) {
           var->value = true;
        } else if (!strcmp(s, "0") ||
                   !strcmp(s, "no") ||
                   !strcmp(s, "off")) {
           var->value = false;
        } else {
           var->value = false;
        }
 	var->set = true;
    } else {
	var->value = val_default;
        var->set = false;
    }
    return var;
}

env_var_double_t *vftr_read_env_double (char *env_name, double val_default) {
    char *s;
    env_var_double_t *var;
    var = (env_var_double_t*)malloc (sizeof (env_var_double_t));
    if (s = getenv (env_name)) {
	var->value = atoi(s);
 	var->set = true;
    } else {
	var->value = val_default;
        var->set = false;
    }
    return var;
}

env_var_string_t *vftr_read_env_string (char *env_name, char *val_default) {
    char *s;
    env_var_string_t *var;
    var = (env_var_string_t*)malloc (sizeof (env_var_string_t));
    if (s = getenv (env_name)) {
	var->value = s;
 	var->set = true;
    } else {
	var->value = val_default;
        var->set = false;
    }
    return var;
}

env_var_regex_t *vftr_read_env_regex (char *env_name, regex_t *val_default) {
    char *s;
    env_var_regex_t *var;
    var = (env_var_regex_t*)malloc (sizeof (env_var_regex_t));
    if (s = getenv (env_name)) {
	var->value = vftr_compile_regexp (s);
 	var->set = true;
    } else {
	var->value = val_default;
        var->set = false;
    }
    return var;
}

void vftr_read_environment () {
    vftr_environment = (vftr_envs_t*)malloc (sizeof (vftr_envs_t));

    vftr_environment->vftrace_off = vftr_read_env_bool ("VFTR_OFF", false);
    vftr_environment->do_sampling = vftr_read_env_bool ("VFTR_SAMPLING", false);
    vftr_environment->regions_precise = vftr_read_env_bool ("VFTR_REGIONS_PRECISE", true);
    vftr_environment->output_directory = vftr_read_env_string ("VFTR_OUT_DIRECTORY", ".");
    vftr_environment->logfile_basename = vftr_read_env_string ("VFTR_LOGFILE_BASENAME", NULL);
    vftr_environment->logfile_all_ranks = vftr_read_env_bool ("VFTR_LOGFILE_ALL_RANKS", false);
    vftr_environment->sampletime = vftr_read_env_double ("VFTR_SAMPLETIME", 0.005);
    vftr_environment->stoptime = vftr_read_env_long_long ("VFTR_STOPTIME", 7ll*24ll*60ll*60ll);
    vftr_environment->accurate_profile = vftr_read_env_bool ("VFTR_ACCURATE_PROFILE", false);
    vftr_environment->prof_truncate = vftr_read_env_bool ("VFTR_PROF_TRUNCATE", true);
    vftr_environment->mpi_log = vftr_read_env_bool ("VFTR_MPI_LOG", false);
    vftr_environment->signals_off = vftr_read_env_bool ("VFTR_SIGNALS_OFF", true);
    vftr_environment->bufsize = vftr_read_env_int ("VFTR_BUFSIZE", 8); 
    vftr_environment->runtime_profile_funcs = vftr_read_env_regex ("VFTR_RUNTIME_PROFILE_FUNCS", NULL);
    vftr_environment->include_only_regex = vftr_read_env_regex ("VFTR_INCLUDE_ONLY", NULL);
    vftr_environment->exclude_functions_regex = vftr_read_env_regex ("VFTR_EXCLUDE", NULL);
    vftr_environment->detail_until_cum_cycles = vftr_read_env_double ("VFTR_DETAIL_UNTIL_CUM_CYCLES", 90.0);
    vftr_environment->scenario_file = vftr_read_env_string ("VFTR_SCENARIO_FILE", NULL);
    vftr_environment->preciseregex = vftr_read_env_regex ("VFTR_PRECISE", NULL);
    vftr_environment->license_verbose = vftr_read_env_bool ("VFTR_LICENSE_VERBOSE", false);
    vftr_environment->print_stacks_for = vftr_read_env_string ("VFTR_PRINT_STACKS_FOR", NULL);
    vftr_environment->print_loadinfo_for = vftr_read_env_string ("VFTR_PRINT_LOADINFO_FOR", NULL);
}

void vftr_assert_environment () {
	assert (vftr_environment->bufsize->value > 0);
	if (vftr_environment->include_only_regex->set && vftr_environment->exclude_functions_regex->set) {
		vftr_environment->exclude_functions_regex->set = false;
	}

	if (vftr_environment->scenario_file->set) {
		FILE *fp;
		if (!(fp = fopen (vftr_environment->scenario_file->value, "r"))) {
			vftr_environment->scenario_file->set = false;
		} else {
			fclose(fp);
		}
	}

	// When neither VEPERF nor PAPI are activated, any given scenario file is ignored
#if !defined(HAS_VEPERF) && !defined(HAS_PAPI)
	vftr_environment->scenario_file->set = false;
#endif

	if (vftr_environment->detail_until_cum_cycles->value > 100.0 || 
	    vftr_environment->detail_until_cum_cycles->value < 0.0) {
		printf ("Warning: Invalid value for VFTR_DETAIL_UNTIL_CUM_CYLES (%lf). Set to default value of 90.0\n",
			vftr_environment->detail_until_cum_cycles->value);
		vftr_environment->detail_until_cum_cycles->value = 90.0;
	}

	// We check whether the number given after the ":", i.e. the group size, is positive.
	// Note that group_size == 0 is not supported, it is to be activated by leaving out the
	// colon instead.
	if (vftr_environment->print_stacks_for->set) {
		char *vftr_print_groups = vftr_environment->print_stacks_for->value;
		while (*vftr_print_groups) {
			int group_base, group_size;
			char *p;
			for (p = vftr_print_groups; *p && *p != ','; p++);
			switch (sscanf (vftr_print_groups, "%d:%d", &group_base, &group_size)) {
			case 2:
				if (group_size <= 0) {
					printf ("Warning: The group size in the environment variable VFTR_PRINT_STACKS_FOR=<group_base>:<group_size>	has to be greater than zero.\n");
					printf ("No additional stack info will be printed.\n");
					vftr_environment->print_stacks_for->set = false;
				}
			}
			vftr_print_groups = *p ? p + 1 : p;
		}		
	}
	// The same for VFTR_PRINT_LOADINFO_FOR
	if (vftr_environment->print_loadinfo_for->set) {
		char *vftr_mpi_groups = vftr_environment->print_loadinfo_for->value;
		while (*vftr_mpi_groups) {
			int group_base, group_size;
			char *p;
			for (p = vftr_mpi_groups; *p && *p != ','; p++);
			switch (sscanf (vftr_mpi_groups, "%d:%d", &group_base, &group_size)) {
			case 2:
				if (group_size <= 0) {
					printf ("Warning: The group size in the environment variable VFTR_PRINT_LOADINFO_FOR=<group_base>:<group_size>	has to be greater than zero.\n");
					printf ("No additional stack info will be printed.\n");
					vftr_environment->print_loadinfo_for->set = false;
				}
			}
			vftr_mpi_groups = *p ? p + 1 : p;
		}		
	}


}

bool vftr_off () {
	return vftr_environment->vftrace_off->value;
}

void vftr_switch_off () {
	vftr_environment->vftrace_off->value = true;
}

bool vftr_env_do_sampling () {
	return vftr_environment->do_sampling->value;
}
