#ifndef SAMPLING_TYPES_H
#define SAMPLING_TYPES_H

#include <stdio.h>
#include <stdbool.h>

typedef struct {
   bool do_sampling;
   char *vfdfilename;
   FILE *vfdfilefp;
   char *iobuffer;
   long long nextsampletime;
   unsigned int function_samplecount;
   unsigned int message_samplecount;
   long long stacktable_offset;
   long long samples_offset;
} sampling_t;

#endif
