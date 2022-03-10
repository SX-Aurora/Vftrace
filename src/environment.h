#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <stdio.h>

#include "environment_types.h"

void vftr_print_env(FILE *fp, environment_t environment);

environment_t vftr_read_environment();

void vftr_environment_free(environment_t *environment_ptr);

#endif
