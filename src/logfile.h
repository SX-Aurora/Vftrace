#ifndef LOGFILE_H
#define LOGFILE_H

#include "environment_types.h"
#include "vftrace_state.h"

char *vftr_get_logfile_name(environment_t environment, int rankID, int nranks);

void vftr_write_logfile(vftrace_t vftrace, long long runtime);

#endif
