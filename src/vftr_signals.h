#ifndef VFTR_SIGNALS_H
#define VFTR_SIGNALS_H

extern int vftr_signal_number;

void vftr_setup_signals();
void *vftr_signal_name(int signum);

#endif
