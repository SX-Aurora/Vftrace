#ifndef LOGFILE_STACKLIST_H
#define LOGFILE_STACKLIST_H

#include <stdlib.h>
#include <stdio.h>

#include "collated_stack_types.h"

void vftr_write_logfile_global_stack_list(FILE *fp, collated_stacktree_t stacktree);

#endif
