#ifndef LOG_PROFILE_H
#define LOG_PROFILE_H

#include "stack_types.h"

int *vftr_stack_calls_list(stacktree_t stacktree);

double *vftr_stack_inclusive_time_list(stacktree_t stacktree);

double *vftr_stack_exclusive_time_list(stacktree_t stacktree);

double *vftr_stack_overhead_time_list(stacktree_t stacktree);

char **vftr_stack_function_name_list(stacktree_t stacktree);

char **vftr_stack_caller_name_list(stacktree_t stacktree);

#endif
