#ifndef _SIGNAL_HANDLING_H
#define _SIGNAL_HANDLING_H

#include <stdlib.h>

void vftr_abort (int errcode);

void vftr_write_signal_message (FILE *fp);

void vftr_setup_signals ();

#endif
