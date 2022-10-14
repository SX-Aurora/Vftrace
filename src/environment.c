#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <limits.h>

#include "self_profile.h"
#include "environment.h"
#include "regular_expressions.h"
#include "range_expand.h"
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
      if (!strcmp(value_string, "1") ||
          !strcmp(value_string, "yes") ||
          !strcmp(value_string, "on")) {
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

void vftr_print_environment(FILE *fp, environment_t environment) {
   if (environment.valid && environment.print_environment.value.bool_val) {
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
   SELF_PROFILE_START_FUNCTION;
   environment_t environment;
   environment.vftrace_off = vftr_read_env_bool("VFTR_OFF", false);
   environment.do_sampling = vftr_read_env_bool("VFTR_SAMPLING", false);
   environment.output_directory = vftr_read_env_string("VFTR_OUT_DIRECTORY", ".");
   environment.logfile_basename = vftr_read_env_string("VFTR_LOGFILE_BASENAME", NULL);
   environment.logfile_for_ranks = vftr_read_env_string("VFTR_LOGFILE_FOR_RANKS", "none");
   environment.ranks_in_mpi_profile = vftr_read_env_string("VFTR_RANKS_IN_MPI_PROFILE", "all");
   environment.sampletime = vftr_read_env_double("VFTR_SAMPLETIME", 0.005);
   environment.mpi_log = vftr_read_env_bool("VFTR_MPI_LOG", false);
   environment.mpi_show_sync_time = vftr_read_env_bool("VFTR_MPI_SHOW_SYNC_TIME", false);
   environment.vfd_bufsize = vftr_read_env_int("VFTR_VFD_BUFSIZE", 8);
   environment.include_only_regex = vftr_read_env_regex("VFTR_INCLUDE_ONLY", NULL);
   environment.scenario_file = vftr_read_env_string("VFTR_SCENARIO_FILE", NULL);
   environment.preciseregex = vftr_read_env_regex("VFTR_PRECISE", NULL);
   environment.print_stack_profile = vftr_read_env_regex("VFTR_PRINT_STACK_PROFILE", NULL);
   environment.print_stacks_for = vftr_read_env_string("VFTR_PRINT_STACKS_FOR", NULL);
   environment.strip_module_names = vftr_read_env_bool("VFTR_STRIP_MODULE_NAMES", false);
   environment.sort_profile_table = vftr_read_env_string("VFTR_SORT_PROFILE_TABLE", "TIME_EXCL");
   environment.sort_mpi_table = vftr_read_env_string("VFTR_SORT_MPI_TABLE", "NONE");
   environment.show_overhead = vftr_read_env_bool("VFTR_SHOW_FUNCTION_OVERHEAD", false);
   environment.show_calltime_imbalances = vftr_read_env_bool("VFTR_SHOW_CALLTIME_IMBALANCES", false);
   environment.group_functions_by_name = vftr_read_env_bool("VFTR_GROUP_FUNCTIONS_BY_NAME", false);
   environment.print_environment = vftr_read_env_bool("VFTR_PRINT_ENVIRONMENT", true);
   environment.callpath_in_profile = vftr_read_env_bool("VFTR_CALLPATH_IN_PROFILE", false);
   environment.callpath_in_mpi_profile = vftr_read_env_bool("VFTR_CALLPATH_IN_MPI_PROFILE", false);
   environment.demangle_cxx = vftr_read_env_bool("VFTR_DEMANGLE_CXX", false);
   environment.nenv_vars = 25;
   environment.valid = true;
   SELF_PROFILE_END_FUNCTION;
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
      if (ld == 0) {
         return;
      }
   }
}

// There might be mistyped Vftrace environment variables. Loop over all existing env variables,
// check if they match a Vftrace variable, and make an alternative suggestion if it is possibly mistyped.
void vftr_check_env_names(FILE *fp, environment_t *environment_ptr) {
   SELF_PROFILE_START_FUNCTION;
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
   SELF_PROFILE_END_FUNCTION;
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
   SELF_PROFILE_START_FUNCTION;
   environment_t environment = *environment_ptr;
   if (environment.valid) {
      environment.valid = false;
      for (int ienv=0; ienv<environment_ptr->nenv_vars; ienv++) {
         vftr_env_var_free(vftr_get_env_var_ptr_by_idx(environment_ptr, ienv));
      }
   }
   SELF_PROFILE_END_FUNCTION;
}

// Attempt to check the user supplied environment values for soundness
void vftr_environment_assert_vftrace_off(FILE *fp, env_var_t vftrace_off) {
   (void) fp;
   (void) vftrace_off;
}

void vftr_environment_assert_do_sampling(FILE *fp, env_var_t do_sampling) {
   (void) fp;
   (void) do_sampling;
}

void vftr_environment_assert_output_directory(FILE *fp, env_var_t output_directory) {
   (void) fp;
   (void) output_directory;
}

void vftr_environment_assert_logfile_basename(FILE *fp, env_var_t logfile_basename) {
   (void) fp;
   (void) logfile_basename;
}

void vftr_environment_assert_logfile_for_ranks(FILE *fp, env_var_t logfile_for_ranks) {
   char *rangelist = logfile_for_ranks.value.string_val;
   if (!strcmp(rangelist, "all")) {
      ;
   } else if (!strcmp(rangelist, "none")) {
      ;
   } else {
      int nvals = 0;
      int *exp_list = vftr_expand_rangelist(rangelist, &nvals);
      if (nvals == 0 || exp_list == NULL) {
         fprintf(fp, "Warning: Unable to properly parse given list for \"%s\".\n",
                 rangelist);
      }
      for (int i=0; i<nvals; i++) {
         if (exp_list[i] < 0) {
            fprintf(fp, "Warning: %s=%s results in negative rank values.",
                    logfile_for_ranks.name, rangelist);
         }
      }
      free(exp_list);
   }
}

void vftr_environment_assert_ranks_in_mpi_profile(FILE *fp,
                                                  env_var_t ranks_in_mpi_profile) {
   char *rangelist = ranks_in_mpi_profile.value.string_val;
   if (strcmp(rangelist, "all")) {
      int nvals = 0;
      int *exp_list = vftr_expand_rangelist(rangelist, &nvals);
      if (nvals == 0 || exp_list == NULL) {
         fprintf(fp, "Warning: Unable to properly parse given list for \"%s\".\n",
                 rangelist);
      }
      for (int i=0; i<nvals; i++) {
         if (exp_list[i] < 0) {
            fprintf(fp, "Warning: %s=%s results in negative rank values.",
                    ranks_in_mpi_profile.name, rangelist);
         }
      }
      free(exp_list);
   }
}

void vftr_environment_assert_sampletime(FILE *fp, env_var_t sampletime) {
   if (sampletime.value.double_val <= 0.0) {
      fprintf(fp, "Warning: \"%s\" with a value smaller or equal to 0 "
              "will sample every event.\n",
              sampletime.name);
   }
}

void vftr_environment_assert_mpi_log(FILE *fp, env_var_t mpi_log) {
   (void) fp;
   (void) mpi_log;
}

void vftr_environment_assert_mpi_show_sync_time(FILE *fp,
                                                env_var_t mpi_show_sync_time) {
   (void) fp;
   (void) mpi_show_sync_time;
}

void vftr_environment_assert_vfd_bufsize(FILE *fp, env_var_t vfd_bufsize) {
   if (vfd_bufsize.value.int_val <= 0) {
      fprintf(fp, "Warning: \"%s\" needs to be > 0, but is %d\n",
              vfd_bufsize.name, vfd_bufsize.value.int_val);
   }
}

void vftr_environment_assert_include_only_regex(FILE *fp, env_var_t include_only_regex) {
   (void) fp;
   (void) include_only_regex;
}

void vftr_environment_assert_scenario_file(FILE *fp, env_var_t scenario_file) {
   if (scenario_file.set) {
      char *scn_file = scenario_file.value.string_val;
      if (strlen(scn_file) == 0) {
         fprintf(fp, "Warning: %s set, but to an empty value.\n",
                 scenario_file.name);
      } else {
         if (access(scn_file, F_OK) != 0) {
            fprintf(fp, "Warning: %s is set to %s, but file could not be found\n",
                    scenario_file.name, scenario_file.value.string_val);
         }
      }
   }
}

void vftr_environment_assert_preciseregex(FILE *fp, env_var_t preciseregex) {
   (void) fp;
   (void) preciseregex;
}

void vftr_environment_assert_print_stack_profile(FILE *fp,
                                                 env_var_t print_stack_profile) {
   (void) fp;
   (void) print_stack_profile;
}

void vftr_environment_assert_print_stacks_for(FILE *fp, env_var_t print_stacks_for) {
   (void) fp;
   (void) print_stacks_for;
}

void vftr_environment_assert_strip_module_names(FILE *fp, env_var_t strip_module_names) {
   (void) fp;
   (void) strip_module_names;
}

void vftr_environment_assert_sort_profile_table(FILE *fp, env_var_t sort_profile_table) {
   char *sort_str = sort_profile_table.value.string_val;
   int nvalid_str = 6;
   char *valid_str[] = {"TIME_EXCL", "TIME_INCL", "CALLS",
                        "STACK_ID", "OVERHEAD", "NONE"};
   bool valid = false;
   for (int istr=0; istr<nvalid_str; istr++) {
      valid = valid || !strcmp(sort_str, valid_str[istr]);
   }
   if (!valid) {
      fprintf(fp, "Warning: %s was set to \"%s\", but only valid options are:",
              sort_profile_table.name, sort_str);
      for (int istr=0; istr<nvalid_str-1; istr++) {
         fprintf(fp, " \"%s\",", valid_str[istr]);
      }
      fprintf(fp, " and \"%s\".\n", valid_str[nvalid_str-1]);
   }
}

void vftr_environment_assert_sort_mpi_table(FILE *fp, env_var_t sort_mpi_table) {
   char *sort_str = sort_mpi_table.value.string_val;
   int nvalid_str = 8;
   char *valid_str[] = {"MESSAGES",
                        "SEND_SIZE", "RECV_SIZE",
                        "SEND_BW", "SEND_BW",
                        "COMM_TIME", "STACK_ID",
                        "NONE"};
   bool valid = false;
   for (int istr=0; istr<nvalid_str; istr++) {
      valid = valid || !strcmp(sort_str, valid_str[istr]);
   }
   if (!valid) {
      fprintf(fp, "Warning: %s was set to \"%s\", but only valid options are:",
              sort_mpi_table.name, sort_str);
      for (int istr=0; istr<nvalid_str-1; istr++) {
         fprintf(fp, " \"%s\",", valid_str[istr]);
      }
      fprintf(fp, " and \"%s\".\n", valid_str[nvalid_str-1]);
   }
}

void vftr_environment_assert_show_overhead(FILE *fp, env_var_t show_overhead) {
   (void) fp;
   (void) show_overhead;
}

void vftr_environment_assert_show_calltime_imbalances(FILE *fp, env_var_t show_calltime_imbalances) {
   (void) fp;
   (void) show_calltime_imbalances;
}

void vftr_environment_assert_group_functions_by_name(FILE* fp, env_var_t group_functions_by_name) {
   (void) fp;
   (void) group_functions_by_name;
}

void vftr_environment_assert_print_environment(FILE *fp, env_var_t print_environment) {
   (void) fp;
   (void) print_environment;
}

void vftr_environment_assert_callpath_in_profile(FILE *fp,
                                                 env_var_t callpath_in_profile) {
   (void) fp;
   (void) callpath_in_profile;
}

void vftr_environment_assert_callpath_in_mpi_profile(FILE *fp,
                                                     env_var_t callpath_in_mpi_profile) {
   (void) fp;
   (void) callpath_in_mpi_profile;
}

void vftr_environment_assert_demangle_cxx(FILE *fp, env_var_t demangle_cxx) {
   (void) fp;
   (void) demangle_cxx;
}

void vftr_environment_assert(FILE *fp, environment_t environment) {
   SELF_PROFILE_START_FUNCTION;
   vftr_environment_assert_vftrace_off(fp, environment.vftrace_off);
   vftr_environment_assert_do_sampling(fp, environment.do_sampling);
   vftr_environment_assert_output_directory(fp, environment.output_directory);
   vftr_environment_assert_logfile_basename(fp, environment.logfile_basename);
   vftr_environment_assert_logfile_for_ranks(fp, environment.logfile_for_ranks);
   vftr_environment_assert_ranks_in_mpi_profile(fp, environment.ranks_in_mpi_profile);
   vftr_environment_assert_sampletime(fp, environment.sampletime);
   vftr_environment_assert_mpi_log(fp, environment.mpi_log);
   vftr_environment_assert_mpi_show_sync_time(fp, environment.mpi_show_sync_time);
   vftr_environment_assert_vfd_bufsize(fp, environment.vfd_bufsize);
   vftr_environment_assert_include_only_regex(fp, environment.include_only_regex);
   vftr_environment_assert_scenario_file(fp, environment.scenario_file);
   vftr_environment_assert_preciseregex(fp, environment.preciseregex);
   vftr_environment_assert_print_stack_profile(fp, environment.print_stack_profile);
   vftr_environment_assert_print_stacks_for(fp, environment.print_stacks_for);
   vftr_environment_assert_strip_module_names(fp, environment.strip_module_names);
   vftr_environment_assert_sort_profile_table(fp, environment.sort_profile_table);
   vftr_environment_assert_sort_mpi_table(fp, environment.sort_mpi_table);
   vftr_environment_assert_show_overhead(fp, environment.show_overhead);
   vftr_environment_assert_show_calltime_imbalances(fp, environment.show_calltime_imbalances);
   vftr_environment_assert_group_functions_by_name(fp, environment.group_functions_by_name);
   vftr_environment_assert_print_environment(fp, environment.print_environment);
   vftr_environment_assert_callpath_in_profile(fp, environment.callpath_in_profile);
   vftr_environment_assert_callpath_in_mpi_profile(fp, environment.callpath_in_mpi_profile);
   vftr_environment_assert_demangle_cxx(fp, environment.demangle_cxx);
   SELF_PROFILE_END_FUNCTION;
}
