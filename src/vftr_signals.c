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

#include "vftr_setup.h"
#include "vftr_signals.h"

void vftr_abort (int errcode) {
#ifdef _MPI
    (void) PMPI_Abort( MPI_COMM_WORLD, errcode );
#else
    abort();
#endif
}

bool vftr_sigterm_called;
bool vftr_sigint_called;

struct sigaction vftr_sigterm; 
struct sigaction vftr_sigint; 

/**********************************************************************/

void vftr_sigterm_handler (int signum) {
  printf ("Vftrace received SIGTERM on rank %d %d\n", vftr_mpirank, vftr_sigterm_called);
  if (!vftr_sigterm_called) {
    vftr_finalize();
    vftr_sigterm.sa_handler = SIG_DFL; 
    sigaction (SIGTERM, &vftr_sigterm, NULL);
    raise(SIGTERM);
    printf ("HUHU\n");
    vftr_sigterm_called = true;
  }
}

void vftr_sigint_handler (int signum) {
  printf ("Vftrace received SIGINT on rank %d\n", vftr_mpirank);
  if (!vftr_sigint_called) {
    vftr_finalize();
    vftr_sigint.sa_handler = SIG_DFL;
    sigaction (SIGINT, &vftr_sigint, NULL);
    raise(SIGINT);
    vftr_sigint_called = true;
  }
}

/**********************************************************************/

void vftr_setup_signals () {

  printf ("Vftrace: Signals have been setup\n");
  memset (&vftr_sigterm, 0, sizeof(vftr_sigterm));
  memset (&vftr_sigint, 0, sizeof(vftr_sigint));
  vftr_sigterm.sa_handler = vftr_sigterm_handler;
  vftr_sigterm.sa_flags = SA_SIGINFO;
  vftr_sigint.sa_handler = vftr_sigint_handler;
  vftr_sigint.sa_flags = SA_SIGINFO;

  sigaction (SIGTERM, &vftr_sigterm, NULL);
  sigaction (SIGINT, &vftr_sigint, NULL);

  vftr_sigterm_called = false;
  vftr_sigint_called = false;
}

/**********************************************************************/
