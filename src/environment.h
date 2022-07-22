#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <stdio.h>

#include "environment_types.h"

env_var_t *vftr_get_env_var_ptr_by_idx(environment_t *environment_ptr, int idx);

void vftr_print_environment(FILE *fp, environment_t environment);

environment_t vftr_read_environment();

void vftr_check_env_names(FILE *fp, environment_t *environment_ptr);

void vftr_environment_free(environment_t *environment_ptr);

void vftr_environment_assert(FILE *fp, environment_t environment);

#endif
