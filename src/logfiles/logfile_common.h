#ifndef LOGFILE_COMMON_H
#define LOGFILE_COMMON_H

#include <stdlib.h>
#include <stdbool.h>

#include "vftrace_state.h"
#include "configuration_types.h"
#include "logfile_common_types.h"

char *vftr_get_logfile_name (config_t config, char *type, int rankID, int nranks);

FILE *vftr_get_this_logfile_fp (char *type, FILE *fp_main, int rankID, int nranks);

vftr_logfile_fp_t vftr_logfile_open_fps (config_t config, int rankID, int nranks);

void vftr_logfile_close_fp (vftr_logfile_fp_t all_fp);

void vftr_write_logfile_warnings (vftrace_t vftrace, vftr_logfile_fp_t all_fp);

void vftr_write_logfile_prologue (bool is_master_logfile, vftrace_t vftrace,
                                  vftr_logfile_fp_t all_fp, long long runtime);
#endif
