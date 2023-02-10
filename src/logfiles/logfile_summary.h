#ifndef LOGFILE_HEADER_H
#define LOGFILE_HEADER_H

#include <stdlib.h>
#include <stdio.h>

#include "size_types.h"
#include "timer_types.h"
#include "process_types.h"

void vftr_write_logfile_summary(FILE *fp, process_t process,
                                vftr_size_t vftrace_size,
                                long long runtime);

#endif
