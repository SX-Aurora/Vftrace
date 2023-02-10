#ifndef LOGFILE_COMMON_TYPES_H
#define LOGFILE_COMMON_TYPES_H

#define N_LOGFILE_TYPES 7

#include <stdio.h>

enum {LOG_MAIN, LOG_MINMAX, LOG_GROUPED,
      LOG_MPI, LOG_CUDA, LOG_ACCPROF, LOG_HWPROF};

typedef struct {
  FILE *fp[N_LOGFILE_TYPES];
} vftr_logfile_fp_t;



#endif
