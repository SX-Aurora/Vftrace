#include <stdlib.h>
#include <time.h>

#include "self_profile.h"

void vftr_microsleep(unsigned long usec) {
   SELF_PROFILE_START_FUNCTION;
   unsigned long sec = usec/1000000ul;
   unsigned long nsec = 1000ul*(usec - sec*1000000ul);
   struct timespec req = {
      .tv_sec = sec,
      .tv_nsec = nsec
   };
   nanosleep(&req, NULL);
   SELF_PROFILE_END_FUNCTION;
}
