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

#include "vftr_signals.h"

void vftr_abort (int errcode) {
#ifdef _MPI
    (void) PMPI_Abort( MPI_COMM_WORLD, errcode );
#else
    abort();
#endif
}

/**********************************************************************/

void vftr_sigterm (int signum, siginfo_t *info, void *ptr) {
  printf ("Vftrace received SIGTERM\n");
}

void vftr_sigint (int signum, siginfo_t *info, void *ptr) {
  printf ("Vftrace received SIGINT\n");
}

/**********************************************************************/

void vftr_setup_signals () {
  struct sigaction s_sigterm; 
  struct sigaction s_sigint; 

  printf ("Vftrace: Signals have been setup\n");
  memset (&s_sigterm, 0, sizeof(s_sigterm));
  memset (&s_sigint, 0, sizeof(s_sigint));
  s_sigterm.sa_sigaction = vftr_sigterm;
  s_sigterm.sa_flags = SA_SIGINFO;
  s_sigint.sa_sigaction = vftr_sigint;
  s_sigint.sa_flags = SA_SIGINFO;

  sigaction (SIGTERM, &s_sigterm, NULL);
  sigaction (SIGINT, &s_sigint, NULL);
}

/**********************************************************************/
