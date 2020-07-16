#include <stdbool.h>
#include <time.h>
#include <pthread.h>
#include "vftr_timer.h"
#include "vftr_monitoring_thread.h"

// seperate pthread to monitor program activities in the background
pthread_t vftr_monitor_thread;

// lock for thread synchronisation
pthread_mutex_t vftr_monitor_thread_lock_handle;

// keep thread alive
// This variable must never be written by any function but
// vftr_create_monitor_thread and vftr_join_monitor_thread
bool vftr_keep_monitor_thread_alive = false;

// argument strucutre for thread
typedef struct {
   long long monitor_interval_usec;
} vftr_monitor_data_t;
vftr_monitor_data_t thread_data;

// monitoring function to be run by a seperate pthread
void *vftr_monitor_thread_fkt(void *arg) {
   // translate variables from arg-struct to individual variables
   vftr_monitor_data_t *thread_argument = (vftr_monitor_data_t*) arg;
   long long monitor_interval = thread_argument->monitor_interval_usec;

   // variables to keep track of time
   long long tstart = vftr_get_runtime_usec();
   long long tend = 0ll;

   // start monitoring loop
   // lock thread for check of keep alive signal
   pthread_mutex_lock(&vftr_monitor_thread_lock_handle);
   while (vftr_keep_monitor_thread_alive) {
      // Calls to shared variables
      

      pthread_mutex_unlock(&vftr_monitor_thread_lock_handle);
      // Calls to local variables


      // No further instructions after this comment!
      //
      // keep idle until the next monitoring time is due
      // take this as end time of iteration
      // assume timer conversion is instant
      tend = vftr_get_runtime_usec();
      long long sleeptime_usec = monitor_interval - (tend - tstart);
      struct timespec sleeptime_spec;
      sleeptime_spec.tv_sec  = sleeptime_usec / 1000000l;
      sleeptime_spec.tv_nsec = 1000 * (sleeptime_usec % 1000000000l);
      // sleep
      nanosleep(&sleeptime_spec, NULL);
      // take this as the start time of the next iteration
      tstart = vftr_get_runtime_usec();
      // lock thread for check of keep alive signal
      pthread_mutex_lock(&vftr_monitor_thread_lock_handle);
   }
   pthread_mutex_unlock(&vftr_monitor_thread_lock_handle);
}

void vftr_create_monitor_thread(long long monitor_interval_usec) {

   // keep the thread alive until the main thread
   // explicitly orders to join threads
   vftr_keep_monitor_thread_alive = true;

   // initialize the mutex lock
   pthread_mutex_init(&vftr_monitor_thread_lock_handle, NULL);

   // prepare input struct for thread function
   thread_data.monitor_interval_usec = monitor_interval_usec;

   // create the monitoring thread
   pthread_create(&vftr_monitor_thread,    // thread 
                  NULL,                    // NULL -> default thread attributes
                  vftr_monitor_thread_fkt, // function to be run as seperate thread
                  &thread_data);           // Input data for the thread
}

void vftr_join_monitor_thread() {
   // signal the thread to terminate on the next monitoring loop
   pthread_mutex_lock(&vftr_monitor_thread_lock_handle);
   vftr_keep_monitor_thread_alive = false;
   pthread_mutex_unlock(&vftr_monitor_thread_lock_handle);

   // join the thread
   // will wait until the thread finished execution
   pthread_join(vftr_monitor_thread, NULL);

   // destroy mutex lock
   pthread_mutex_destroy(&vftr_monitor_thread_lock_handle);
}

void vftr_lock_monitoring_thread() {
   pthread_mutex_lock(&vftr_monitor_thread_lock_handle);
}

void vftr_unlock_monitor_thread() {
   pthread_mutex_unlock(&vftr_monitor_thread_lock_handle);
}
