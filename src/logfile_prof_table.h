#ifndef LOGFILE_PROF_TABLE_H
#define LOGFILE_PROF_TABLE_H

#include <stdlib.h>
#include <stdio.h>

#include "stack_types.h"
#include "environment_types.h"

void vftr_write_logfile_profile_table(FILE *fp, stacktree_t stacktree,
                                      environment_t environment); 

#endif
