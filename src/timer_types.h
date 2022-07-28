#ifndef TIMER_TYPES_H
#define TIMER_TYPES_H

#include <stdbool.h>

#include "time.h"

typedef struct {
   bool valid;
   struct timespec timestamp;
} reftime_t;

typedef struct {
   char *start_time;
   char *end_time;
} time_strings_t;

#endif
