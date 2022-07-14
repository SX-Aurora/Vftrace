#ifndef LOGFILE_HEADER_H
#define LOGFILE_HEADER_H

#include <stdlib.h>
#include <stdio.h>

#include "timer_types.h"
#include "process_types.h"

void vftr_write_logfile_header(FILE *fp, time_strings_t timestrings);

void vftr_write_logfile_summary(FILE *fp, process_t process, long long runtime);

#endif
