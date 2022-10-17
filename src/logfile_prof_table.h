#ifndef LOGFILE_PROF_TABLE_H
#define LOGFILE_PROF_TABLE_H

#include <stdlib.h>
#include <stdio.h>

#include "collated_stack_types.h"
#include "environment_types.h"

void vftr_write_logfile_profile_table(FILE *fp, collated_stacktree_t stacktree,
                                      environment_t environment);

void vftr_write_logfile_name_grouped_profile_table(FILE *fp, collated_stacktree_t stacktree,
                                                   environment_t environment);

#endif
