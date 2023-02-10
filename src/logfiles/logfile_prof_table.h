#ifndef LOGFILE_PROF_TABLE_H
#define LOGFILE_PROF_TABLE_H

#include <stdlib.h>
#include <stdio.h>

#include "collated_stack_types.h"
#include "configuration_types.h"

void vftr_write_logfile_profile_table(FILE *fp,
                                      collated_stacktree_t stacktree,
                                      config_t config);

void vftr_write_logfile_name_grouped_profile_table(FILE *fp,
                                                   collated_stacktree_t stacktree,
                                                   config_t config);

#endif
