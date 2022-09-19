#ifndef TIMER_H
#define TIMER_H

#include <time.h>

long long time_diff_nsec(struct timespec t_start, struct timespec t_end);

#endif
