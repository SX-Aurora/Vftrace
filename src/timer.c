#include <stdlib.h>
#include <stdio.h>

#include <time.h>
#include <string.h>

#include "timer_types.h"

// reference time
// One of the very few global variables
// that is not part of the vftrace state construct
reftime_t vftr_reference_time = {
   .valid = false,
   .timestamp = {0ll,0ll},
};

// CLOCK_MONOTONIC is not affected by NTP or system time changes.
struct timespec vftr_get_timestamp() {
   struct timespec timestamp;
   clock_gettime(CLOCK_MONOTONIC, &timestamp);
   return timestamp;
}

// get the current time in nano seconds since
// the reference time point
long long vftr_get_runtime_nsec() {
   // get the current time
   struct timespec timestamp = vftr_get_timestamp();

   // compute the time difference in nanoseconds
   // difference in second counter
   long long delta_sec  = timestamp.tv_sec  - vftr_reference_time.timestamp.tv_sec;
   // difference in nanosecond counter
   long long delta_nsec = timestamp.tv_nsec - vftr_reference_time.timestamp.tv_nsec;
   // handle a possible carry over of nanoseconds
   if (delta_nsec < 0) {
      // add one second in nano seconds to the nano second difference
      delta_nsec += 1000000000l;
      // substract one second from the second difference 
      delta_sec -= 1l;
   }
   // return the amount of microseconds since the local reference time
   return 1000000000l*delta_sec + delta_nsec;
}

//// get the number of elapsed clock counts
//unsigned long long vftr_get_cycles() {
//   unsigned long long cycles = 0ull;
//#if defined (__ve__)
//// On SX Aurora, we obtain the number of elapsed cycles by reading the usrcc register
//// using the smir command. In earlier versions, lhm.l was used. However, this yields
//// VH clock counts, which is not the correct measure to compute e.g. the vector time.
//   asm volatile ("smir %0, %usrcc" : "=r"(cycles));
//#elif defined (__x86_64__)
//      unsigned int a, d;
//      asm volatile("rdtsc" : "=a" (a), "=d" (d));
//      cycles = ((unsigned long long)a) | (((unsigned long long)d) << 32);
//#endif
//   return cycles;
//}

// set the local reference time to which all
// timedifferences are measured
void vftr_set_local_ref_time() {
   reftime_t ref_timer;
   ref_timer.timestamp = vftr_get_timestamp();
   ref_timer.valid = true;
   vftr_reference_time = ref_timer;
}

char *vftr_get_date_str() {
   time_t current_time = time(NULL);
   char *current_time_string = ctime(&current_time);
   // remove the linebreak at the end of the string
   if (current_time_string != NULL) {
      char *timestr = strdup(current_time_string);
      char *s = timestr;
      while (*s != '\0') {
         *s = (*s == '\n' ? '\0' : *s);
         s++;
      }
      return timestr;
   }
   return NULL;
}

void vftr_timestrings_free(time_strings_t *timestrings) {
   if (timestrings->start_time != NULL) {
      free(timestrings->start_time);
      timestrings->start_time = NULL;
   }
   if (timestrings->end_time != NULL) {
      free(timestrings->end_time);
      timestrings->end_time = NULL;
   }
}

void vftr_print_date_str(FILE *fp) {
   char *timestr = vftr_get_date_str();
   fprintf(fp, "%s", timestr);
   free(timestr);
}
