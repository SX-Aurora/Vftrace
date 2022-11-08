#ifndef LOGFILE_H
#define LOGFILE_H

#include "vftrace_state.h"
#include "configuration_types.h"
#include "process_types.h"

char *vftr_get_logfile_name(config_t config);

void vftr_write_logfile_summary(FILE *fp, process_t process, long long runtime);

void vftr_write_logfile(vftrace_t vftrace, long long runtime);

#endif
