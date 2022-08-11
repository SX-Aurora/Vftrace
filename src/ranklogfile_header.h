#ifndef RANKLOGFILE_HEADER_H
#define RANKLOGFILE_HEADER_H

#include <stdlib.h>
#include <stdio.h>

#include "timer_types.h"
#include "process_types.h"

void vftr_write_ranklogfile_header(FILE *fp, time_strings_t timestrings);

void vftr_write_ranklogfile_summary(FILE *fp, process_t process,
                                    unsigned long long vftrace_size,
                                    long long runtime);

#endif
