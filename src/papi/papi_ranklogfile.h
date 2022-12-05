#ifndef PAPI_RANKLOGFILE_H
#define PAPI_RANKLOGFILE_H

#include <stdio.h>

#include "configuration_types.h"
#include "stack_types.h"

void vftr_write_ranklogfile_papi_table (FILE *fp, stacktree_t stacktree, config_t config);

#endif
