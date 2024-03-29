#include <stdlib.h>
#include <string.h>
#include <signal.h>
#ifdef _MPI
#include <mpi.h>
#endif

#include "vftrace_state.h"
#include "vftr_finalize.h"
#include "signal_handling.h"

void vftr_abort (int errcode) {
#ifdef _MPI
   int mpi_initialized;
   PMPI_Initialized(&mpi_initialized);
   if (mpi_initialized) {
      PMPI_Abort (MPI_COMM_WORLD, errcode);
   } else {
      abort();
   }
#else
   abort();
#endif
}

void vftr_write_signal_message (FILE *fp) {
   fprintf (fp, "\n");
   fprintf (fp, "**************************\n");
   fprintf (fp, "Application was cancelled by signal: %s\n", strsignal(vftrace.signal_received));
   fprintf (fp, "**************************\n");
   fprintf (fp, "\n");
}

void vftr_signal_handler (int signum) {
    vftrace.signal_received = signum;
    vftr_finalize();
    vftrace.signals[SIGTERM].sa_handler = SIG_DFL;
    sigaction(SIGTERM, &(vftrace.signals[SIGTERM]), NULL);
    raise(SIGTERM);
}

void vftr_setup_signal (int signum) {
  memset (&vftrace.signals[signum], 0, sizeof(vftrace.signals[signum]));
  vftrace.signals[signum].sa_handler = vftr_signal_handler;
  vftrace.signals[signum].sa_flags = SA_SIGINFO;
  sigaction (signum, &(vftrace.signals[signum]), NULL);
}

void vftr_setup_signals () {
  vftr_setup_signal (SIGTERM);
  vftr_setup_signal (SIGINT);
  vftr_setup_signal (SIGABRT);
  vftr_setup_signal (SIGFPE);
  vftr_setup_signal (SIGQUIT);
  vftr_setup_signal (SIGSEGV);
}
