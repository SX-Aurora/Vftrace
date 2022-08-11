#ifndef RANKLOGFILE_STACKLIST_H
#define RANKLOGFILE_STACKLIST_H

#include <stdlib.h>
#include <stdio.h>

#include "collated_stack_types.h"

void vftr_write_ranklogfile_global_stack_list(FILE *fp, collated_stacktree_t stacktree);

#endif
