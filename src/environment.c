#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <string.h>
#include <ctype.h>
#include <limits.h>

#include "environment.h"
#include "regular_expressions.h"
#include "misc_utils.h"

env_var_t *vftr_get_env_var_ptr_by_idx(environment_t *environment_ptr, int idx) {
   env_var_t *environment_arr = (env_var_t*) environment_ptr;
   if (idx >= 0 && idx < environment_ptr->nenv_vars) {
      return environment_arr+idx;
   } else {
      return NULL;
   }
}

env_var_t vftr_read_env_int(char *env_name, int default_val) {
   env_var_t env_var;
   env_var.value_kind = env_int;
   env_var.name = env_name;
   env_var.value_string = getenv(env_name);
   if (env_var.value_string != NULL) {
      env_var.value.int_val = atoi(env_var.value_string);
      env_var.set = true;
   } else {
      env_var.value.int_val = default_val;
      env_var.set = false;
   }
   return env_var;
}

env_var_t vftr_read_env_long(char *env_name, long default_val) {
   env_var_t env_var;
   env_var.value_kind = env_long;
   env_var.name = env_name;
   env_var.value_string = getenv(env_name);
   if (env_var.value_string != NULL) {
      env_var.value.long_val = atol(env_var.value_string);
      env_var.set = true;
   } else {
      env_var.value.long_val = default_val;
      env_var.set = false;
   }
   return env_var;
}

env_var_t vftr_read_env_longlong(char *env_name, long long default_val) {
   env_var_t env_var;
   env_var.value_kind = env_longlong;
   env_var.name = env_name;
   env_var.value_string = getenv(env_name);
   if (env_var.value_string != NULL) {
      env_var.value.longlong_val = atoll(env_var.value_string);
      env_var.set = true;
   } else {
      env_var.value.longlong_val = default_val;
      env_var.set = false;
   }
   return env_var;
}

env_var_t vftr_read_env_float(char *env_name, float default_val) {
   env_var_t env_var;
   env_var.value_kind = env_float;
   env_var.name = env_name;
   env_var.value_string = getenv(env_name);
   if (env_var.value_string != NULL) {
      sscanf(env_var.value_string, "%f", &(env_var.value.float_val));
      env_var.set = true;
   } else {
      env_var.value.float_val = default_val;
      env_var.set = false;
   }
   return env_var;
}

env_var_t vftr_read_env_double(char *env_name, double default_val) {
   env_var_t env_var;
   env_var.value_kind = env_double;
   env_var.name = env_name;
   env_var.value_string = getenv(env_name);
   if (env_var.value_string != NULL) {
      sscanf(env_var.value_string, "%lf", &(env_var.value.double_val));
      env_var.set = true;
   } else {
      env_var.value.double_val = default_val;
      env_var.set = false;
   }
   return env_var;
}

env_var_t vftr_read_env_bool(char *env_name, bool default_val) {
   env_var_t env_var;
   env_var.value_kind = env_bool;
   env_var.name = env_name;
   env_var.value_string = getenv(env_name);
   if (env_var.value_string != NULL) {
      char *value_string = strdup(env_var.value_string);
      char *s = value_string;
      while (*s != '\0') {
         *s = tolower(*s);
         s++;
      }
      if (!strcmp(env_var.value_string, "1") ||
          !strcmp(env_var.value_string, "yes") ||
          !strcmp(env_var.value_string, "on")) {
         env_var.value.bool_val = true;
      } else {
         env_var.value.bool_val = false;
      }
      env_var.set = true;
      free(value_string);
   } else {
      env_var.value.bool_val = default_val;
      env_var.set = false;
   }
   return env_var;
}

env_var_t vftr_read_env_string(char *env_name, char *default_val) {
   env_var_t env_var;
   env_var.value_kind = env_string;
   env_var.name = env_name;
   env_var.value_string = getenv(env_name);
   if (env_var.value_string != NULL) {
      env_var.value.string_val = env_var.value_string;
      env_var.set = true;
   } else {
      env_var.value.string_val = default_val;
      env_var.set = false;
   }
   return env_var;
}

env_var_t vftr_read_env_regex(char *env_name, regex_t *default_val) {
   env_var_t env_var;
   env_var.value_kind = env_regex;
   env_var.name = env_name;
   env_var.value_string = getenv(env_name);
   if (env_var.value_string != NULL) {
      env_var.value.regex_val = vftr_compile_regexp(env_var.value_string);
      env_var.set = true;
   } else {
      env_var.value.regex_val = default_val;
      env_var.set = false;
   }
   return env_var;
}

void vftr_print_env_var(FILE *fp, env_var_t env_var) {
   switch (env_var.value_kind) {
      case env_int:
         fprintf(fp, "%s: %d%s\n",
                 env_var.name,
                 env_var.value.int_val,
                 env_var.set ? "" : "(default)");
         break;
      case env_long:
         fprintf(fp, "%s: %ld%s\n",
                 env_var.name,
                 env_var.value.long_val,
                 env_var.set ? "" : "(default)");
         break;
      case env_longlong:
         fprintf(fp, "%s: %lld%s\n",
                 env_var.name,
                 env_var.value.longlong_val,
                 env_var.set ? "" : "(default)");
         break;
      case env_float:
         fprintf(fp, "%s: %f%s\n",
                 env_var.name,
                 env_var.value.float_val,
                 env_var.set ? "" : "(default)");
         break;
      case env_double:
         fprintf(fp, "%s: %lf%s\n",
                 env_var.name,
                 env_var.value.double_val,
                 env_var.set ? "" : "(default)");
         break;
      case env_bool:
         fprintf(fp, "%s: %s%s\n",
                 env_var.name,
                 env_var.value.bool_val ? "true" : "false",
                 env_var.set ? "" : "(default)");
         break;
      case env_string:
         fprintf(fp, "%s: %s%s\n",
                 env_var.name,
                 env_var.value.string_val,
                 env_var.set ? "" : "(default)");
         break;
      case env_regex:
         fprintf(fp, "%s: %s%s\n",
                 env_var.name,
                 env_var.value_string,
                 env_var.set ? "" : "(default)");
         break;
      case env_none:
      default:
         fprintf(fp, "Invalid environment variable type\n");
         break;
   }
}

void vftr_print_env(FILE *fp, environment_t environment) {
   if (environment.valid) {
      for (int ienv=0; ienv<environment.nenv_vars; ienv++) {
         vftr_print_env_var(fp,
            *vftr_get_env_var_ptr_by_idx(&environment, ienv));
      }
   } else {
      fprintf(fp, "Environment is invalid!\n"
                  "Not read yet, or already freed!\n");
   }
}

environment_t vftr_read_environment() {
   environment_t environment;
   environment.vftrace_off = vftr_read_env_bool("VFTR_OFF", false);
   environment.do_sampling = vftr_read_env_bool("VFTR_SAMPLING", false);
   environment.regions_precise = vftr_read_env_bool("VFTR_REGIONS_PRECISE", true);
   environment.output_directory = vftr_read_env_string("VFTR_OUT_DIRECTORY", ".");
   environment.logfile_basename = vftr_read_env_string("VFTR_LOGFILE_BASENAME", NULL);
   environment.logfile_for_ranks = vftr_read_env_string("VFTR_LOGFILE_FOR_RANKS", "0");
   environment.mpi_summary_for_ranks = vftr_read_env_string("VFTR_MPI_SUMMARY_FOR_RANKS", "");
   environment.sampletime = vftr_read_env_double("VFTR_SAMPLETIME", 0.005);
   environment.stoptime = vftr_read_env_longlong("VFTR_STOPTIME", 7ll*24ll*60ll*60ll);
   environment.accurate_profile = vftr_read_env_bool("VFTR_ACCURATE_PROFILE", false);
   environment.prof_truncate = vftr_read_env_bool("VFTR_PROF_TRUNCATE", true);
   environment.prof_truncate_cutoff = vftr_read_env_double("VFTR_PROF_TRUNCATE_CUTOFF", 98.0);
   environment.mpi_log = vftr_read_env_bool("VFTR_MPI_LOG", false);
   environment.mpi_show_sync_time = vftr_read_env_bool("VFTR_MPI_SHOW_SYNC_TIME", false);
   environment.signals_off = vftr_read_env_bool("VFTR_SIGNALS_OFF", true);
   environment.bufsize = vftr_read_env_int("VFTR_BUFSIZE", 8);
   environment.runtime_profile_funcs = vftr_read_env_regex("VFTR_RUNTIME_PROFILE_FUNCS", NULL);
   environment.include_only_regex = vftr_read_env_regex("VFTR_INCLUDE_ONLY", NULL);
   environment.detail_until_cum_cycles = vftr_read_env_double("VFTR_DETAIL_UNTIL_CUM_CYCLES", 90.0);
   environment.scenario_file = vftr_read_env_string("VFTR_SCENARIO_FILE", NULL);
   environment.preciseregex = vftr_read_env_regex("VFTR_PRECISE", NULL);
   environment.print_stack_profile = vftr_read_env_regex("VFTR_PRINT_STACK_PROFILE", NULL);
   environment.license_verbose = vftr_read_env_bool("VFTR_LICENSE_VERBOSE", false);
   environment.print_stacks_for = vftr_read_env_string("VFTR_PRINT_STACKS_FOR", NULL);
   environment.print_loadinfo_for = vftr_read_env_string("VFTR_PRINT_LOADINFO_FOR", NULL);
   environment.strip_module_names = vftr_read_env_bool("VFTR_STRIP_MODULE_NAMES", false);
   environment.create_html = vftr_read_env_bool("VFTR_CREATE_HTML", false);
   environment.sort_profile_table = vftr_read_env_string("VFTR_SORT_PROFILE_TABLE", "EXCL_TIME");
   environment.show_overhead = vftr_read_env_bool("VFTR_SHOW_FUNCTION_OVERHEAD", false);
   environment.meminfo_method = vftr_read_env_string("VFTR_MEMINFO_METHOD", "");
   environment.meminfo_stepsize = vftr_read_env_int("VFTR_MEMINFO_STEPSIZE", 1000);
   environment.print_env = vftr_read_env_bool("VFTR_PRINT_ENVIRONMENT", false);
   environment.no_memtrace = vftr_read_env_bool("VFTR_NO_MEMTRACE", false);
   environment.show_stacks_in_profile = vftr_read_env_bool("VFTR_SHOW_STACKS_IN_PROFILE", false);
   environment.no_stack_normalization = vftr_read_env_bool("VFTR_NO_STACK_NORM", false);
   environment.demangle_cpp = vftr_read_env_bool("VFTR_DEMANGLE_CPP", false);
   environment.show_startup = vftr_read_env_bool("VFTR_SHOW_STARTUP", false);
   environment.nenv_vars = 37;
   environment.valid = true;

#if _DEBUG
   fprintf(stderr, "Vftrace-Environment:\n");
   vftr_print_env(stderr, environment);
#endif

   return environment;
}

// Loop over all Vftrace environment variables. When LD is zero, we have an exact match
// and we exit the subroutine to save time.
void vftr_env_var_find_best_match(environment_t *environment_ptr,
                                  char *var_name, int *best_ld,
                                  int *best_idx) {
   *best_ld = INT_MAX;
   *best_idx = -1;
   for (int ivar=0; ivar<environment_ptr->nenv_vars; ivar++) {
      env_var_t *env_var = vftr_get_env_var_ptr_by_idx(environment_ptr, ivar);
      int ld = vftr_levenshtein_distance(var_name,
                                         env_var->name);
      if (ld < *best_ld) {
         *best_ld = ld;
         *best_idx = ivar;
      }
      if (ld == 0) return;
   }
}

// There might be mistyped Vftrace environment variables. Loop over all existing env variables,
// check if they match a Vftrace variable, and make an alternative suggestion if it is possibly mistyped.
void vftr_check_env_names(FILE *fp, environment_t *environment_ptr) {
   extern char **environ;
   char **s = environ;
   for (; *s; s++) {
      if (strstr(*s, "VFTR_")) {
         // There has to be strdup of s, because strtok modifies the first argument. This
         // points to the external environ, which is better not touched!
         char *tmpstr = strdup(*s);
         char *var_name = strtok(tmpstr, "=");
         int best_ld, best_idx;
         vftr_env_var_find_best_match(environment_ptr, var_name, &best_ld, &best_idx);
         // best_ld == 0 -> Exact match.
         if (best_ld > 0)  {
            env_var_t *env_var = vftr_get_env_var_ptr_by_idx(environment_ptr, best_idx);
            fprintf(fp, "Vftrace environment variable %s not known. Do you mean %s?\n",
                    var_name, env_var->name);
         }
         free(tmpstr);
      }
   }
}

void vftr_env_var_free(env_var_t *env_var_ptr) {
   env_var_t env_var = *env_var_ptr;
   if (env_var_ptr != NULL) {
      switch (env_var.value_kind) {
         case env_int:
            break;
         case env_long:
            break;
         case env_longlong:
            break;
         case env_float:
            break;
         case env_double:
            break;
         case env_bool:
            break;
         case env_string:
            break;
         case env_regex:
            if (env_var.set) {
               regfree(env_var.value.regex_val);
               free(env_var.value.regex_val);
               env_var.value.regex_val = NULL;
            }
            break;
         default:
            break;
      }
   }
}

void vftr_environment_free(environment_t *environment_ptr) {
   environment_t environment = *environment_ptr;
   if (environment.valid) {
      environment.valid = false;
      for (int ienv=0; ienv<environment_ptr->nenv_vars; ienv++) {
         vftr_env_var_free(vftr_get_env_var_ptr_by_idx(environment_ptr, ienv));
      }
   }
}
