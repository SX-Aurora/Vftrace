#ifndef RANKLOGFILE_H
#define RANKLOGFILE_H

#include "vftrace_state.h"
#include "environment_types.h"
#include "process_types.h"

char *vftr_get_ranklogfile_name(environment_t environment, int rankID, int nranks);

void vftr_write_ranklogfile_summary(FILE *fp, process_t process, long long runtime);

void vftr_write_ranklogfile(vftrace_t vftrace, long long runtime);

#endif
