#ifndef TIMER_H
#define TIMER_H

#include <time.h>

long long time_diff_usec(struct timespec t_start, struct timespec t_end);

#endif
