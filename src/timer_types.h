#ifndef TIMER_TYPES_H
#define TIMER_TYPES_H

#include <stdbool.h>

#include "time.h"

typedef struct {
   bool valid;
   struct timespec timestamp;
   unsigned long long cyclecount;
} reftime_t;

#endif
