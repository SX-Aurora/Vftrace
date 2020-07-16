#include <stdbool.h>
#include <pthread.h>

// lock for thread synchronisation
extern pthread_mutex_t vftr_lock_monitor_thread;

void vftr_create_monitor_thread(long long monitor_interval_usec);

void vftr_join_monitor_thread();

// Use these functions to lock the monitoring thread for access on shared data
void vftr_lock_monitoring_thread();
void vftr_unlock_monitor_thread();

