#include <time.h>

long long time_diff_usec(struct timespec t_start, struct timespec t_end) {
   // compute the time difference in microseconds
   // difference in second counter
   long long delta_sec  = t_end.tv_sec  - t_start.tv_sec;
   // difference in nanosecond counter
   long long delta_nsec = t_end.tv_nsec - t_start.tv_nsec;
   // handle a possible carry over of nanoseconds
   if (delta_nsec < 0) {
      // add one second in nano seconds to the nano second difference
      delta_nsec += 1000000000l;
      // substract one second in microseconds from microsecond difference
      delta_sec -= 1l;
   }
   // return the amount of microseconds since the local reference time
   return 1000000l*delta_sec + delta_nsec/1000l;
}
