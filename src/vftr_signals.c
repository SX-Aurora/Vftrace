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
#include <unistd.h>
#include <signal.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>

#include "vftr_hooks.h"
#include "vftr_setup.h"
#include "vftr_signals.h"
#include "vftr_stacks.h"
#include "vftr_filewrite.h"

void vftr_abort (int errcode) {
#ifdef _MPI
    PMPI_Abort (MPI_COMM_WORLD, errcode);
#else
    abort();
#endif
}

int vftr_signal_number;

struct sigaction vftr_signals[NSIG];

/**********************************************************************/

void vftr_signal_handler (int signum) {
  if (vftr_profile_wanted) {
    fprintf (vftr_log, "\n");
    fprintf (vftr_log, "**************************\n");
    fprintf (vftr_log, "Application was cancelled: %s\n", strsignal(signum));
    fprintf (vftr_log, "Head of function stack: %s\n", vftr_fstack->name);
    fprintf (vftr_log, "Note: Stacks not normalized\n");
    fprintf (vftr_log, "**************************\n");
    fprintf (vftr_log, "\n");
  }
  vftr_finalize(false);
  vftr_signals[SIGTERM].sa_handler = SIG_DFL;
  sigaction (SIGTERM, &(vftr_signals[SIGTERM]), NULL);
  int ret = raise(signum);
}

/**********************************************************************/

void vftr_setup_signal (int signum) {
  memset (&vftr_signals[signum], 0, sizeof(vftr_signals[signum]));
  vftr_signals[signum].sa_handler = vftr_signal_handler;
  vftr_signals[signum].sa_flags = SA_SIGINFO;
  sigaction (signum, &(vftr_signals[signum]), NULL);
}

void vftr_setup_signals () {

  vftr_signal_number = -1;

  vftr_setup_signal (SIGTERM);
  vftr_setup_signal (SIGINT);
  vftr_setup_signal (SIGABRT);
  vftr_setup_signal (SIGFPE);
  vftr_setup_signal (SIGQUIT);
  vftr_setup_signal (SIGSEGV);
}

/**********************************************************************/
