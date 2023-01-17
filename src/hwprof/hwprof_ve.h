#ifndef HWPROF_VE_H
#define HWPROF_VE_H

#define VE_MAX_HWC_EVENTS 16
//const char *vftr_sx_hwc_names[VE_MAX_HWC_EVENTS] = {"EX", "VX", "FPEC", "VE", "VECC", "L1MCC",
//        "VE2", "VAREC", "VLDEC", "PCCC", "VLPC", "VLEC", "VLCME", "FMAEC", "PTCC", "TTCC"};

void vftr_veprof_init ();
int vftr_ve_counter_index (char *hwc_name);

long long *vftr_get_all_ve_counters();
long long *vftr_get_active_ve_counters();

#endif
