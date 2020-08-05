/*
   This file is part of Vftrace.

   Vftrace is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   Vftrace is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#define _GNU_SOURCE

#ifdef _MPI
#include <mpi.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>

#include "vftr_hwcounters.h"
#include "vftr_signals.h"
#include "vftr_omp.h"
#include "vftr_setup.h"
#include "vftr_stacks.h"
#include "vftr_filewrite.h"
#include "vftr_timer.h"

struct sigaction vftr_old_action[NSIG];
struct sigaction vftr_usr_action[NSIG];

int vftr_signal_number = 0;

/**********************************************************************/

void vftr_abort (int errcode) {
#ifdef _MPI
    (void) PMPI_Abort( MPI_COMM_WORLD, errcode );
#else
    abort();
#endif
}

/**********************************************************************/

void vftr_define_signal_handlers () {
  vftr_sigaction (SIGSEGV);
  vftr_sigaction (SIGBUS);
  vftr_sigaction (SIGFPE);
  vftr_sigaction (SIGTERM);
  vftr_sigaction (SIGINT);
  vftr_sigaction (SIGUSR1);
}

/**********************************************************************/

void vftr_signal (int sig) {
  vftr_signal_number = sig;

  fprintf (vftr_log, "vftr_signal: %s [SIGNUM=%d] - "
                   "closing trace file\n", strsignal(sig), sig);

  vftr_finalize ();

  if (sig == SIGUSR1) {
      vftr_signal_number = 0;
      return;
  }

  sigaction (sig, &vftr_old_action[sig], NULL); /* Restore old handler */
#ifdef _MPI
  if (vftr_mpisize > 1) {
    fprintf (vftr_log, "vftr_signal: calling vftr_abort()\n");
    vftr_abort( sig );
  }
#endif
  fprintf (vftr_log, "vftr_signal: exiting\n");
  exit (1);
}

/**********************************************************************/

/* Establish the signal handlers */

void vftr_sigaction (int sig) {
  sigset_t mask;
  
  sigfillset (&mask);
  vftr_usr_action[sig].sa_handler = vftr_signal;
  vftr_usr_action[sig].sa_mask    = mask;
  vftr_usr_action[sig].sa_flags   = 0;
  
  sigaction (sig, &vftr_usr_action[sig], &vftr_old_action[sig]);
}

/**********************************************************************/

/*
** Interrupt service routine to read and reset the event counters
** before these wrap around. This was a work around for a known
** problem with the HPCM library (no longer supported).
*/
void vftr_sigalarm (int sig) {
   int me = OMP_GET_THREAD_NUM;
   long long time0  = vftr_get_runtime_usec ();
   // get the time to estimate vftrace overhead
   long long overhead_time_start = vftr_get_runtime_usec();

   if (!vftr_timer_end) {
	signal (SIGALRM, vftr_sigalarm);
   }

   int usec, sec;
   if (sig == 0) {
      double sampletime = 0.1;
      usec = (int)(sampletime * 1000000.) % 1000000;
      sec  = (int)(sampletime * 1000000.) / 1000000;
   } else {
      vftr_read_counters (NULL, me);
   }

   if (!vftr_timer_end) {
      struct itimerval timerval;
      timerval.it_interval.tv_sec = 0;
      timerval.it_interval.tv_usec = 0;
      timerval.it_value.tv_sec = sec;
      timerval.it_value.tv_usec = usec;
      setitimer( ITIMER_REAL, &timerval, 0 );
   }

   /* Compensate interrupt overhead in profile */
   long long ohead = vftr_get_runtime_usec () - time0;
   vftr_prof_data[me].cycles += ohead;

   // get the time to estimate vftrace overhead
   long long overhead_time_end = vftr_get_runtime_usec();
   long long overhead = overhead_time_end - overhead_time_start;
   vftr_prof_data[me].timeExcl += overhead;
   vftr_overhead_usec += overhead;
}
