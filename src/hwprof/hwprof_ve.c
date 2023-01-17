#include <string.h>

#include "vftrace_state.h"
#include "hwprof_ve.h"

int vftr_ve_counter_index (char *hwc_name) {
   for (int i = 0; i < VE_MAX_HWC_EVENTS; i++) {
      if (!strcmp (hwc_name, vftrace.hwprof_state.veprof.ve_hwc_names[i])) return i;
   }
   return -1;
}

void vftr_veprof_init () {
   veprof_state_t *state = &(vftrace.hwprof_state.veprof);

   state->ve_hwc_names = (const char**)malloc(VE_MAX_HWC_EVENTS * sizeof(const char*));
   state->ve_hwc_names[0] =  "EX";
   state->ve_hwc_names[1] =  "VX";
   state->ve_hwc_names[2] =  "FPEC";
   state->ve_hwc_names[3] =  "VE";
   state->ve_hwc_names[4] =  "VECC";
   state->ve_hwc_names[5] =  "L1MCC";
   state->ve_hwc_names[6] =  "VE2";
   state->ve_hwc_names[7] =  "VAREC";
   state->ve_hwc_names[8] =  "VLDEC";
   state->ve_hwc_names[9] =  "PCCC";
   state->ve_hwc_names[10] = "VLPC";
   state->ve_hwc_names[11] = "VLEC";
   state->ve_hwc_names[12] = "VLCME";
   state->ve_hwc_names[13] = "FMAEC";
   state->ve_hwc_names[14] = "PTCC";
   state->ve_hwc_names[15] = "TTCC"; 

   state->active_counters = (int*)malloc(VE_MAX_HWC_EVENTS * sizeof(int));  
   memset (state->active_counters, 0, VE_MAX_HWC_EVENTS * sizeof(int));

   for (int i = 0; i < vftrace.hwprof_state.n_counters; i++) {
      char *hwc_name = vftrace.hwprof_state.counters[i].name;
      // The idx range has been asserted beforehand in config_assert
      state->active_counters[vftr_ve_counter_index (hwc_name)] = 1;
   } 
}

void vftr_read_sxhwc_registers (long long hwc[VE_MAX_HWC_EVENTS]) {
    long long tmp[VE_MAX_HWC_EVENTS];
    asm volatile (
        "smir %0,  %%pmc0\n\t"
        "smir %1,  %%pmc1\n\t"
        "smir %2,  %%pmc2\n\t"
        "smir %3,  %%pmc3\n\t"
        "smir %4,  %%pmc4\n\t"
        "smir %5,  %%pmc5\n\t"
        "smir %6,  %%pmc6\n\t"
        "smir %7,  %%pmc7\n\t"
        "smir %8,  %%pmc8\n\t"
        "smir %9,  %%pmc9\n\t"
        "smir %10, %%pmc10\n\t"
        "smir %11, %%pmc11\n\t"
        "smir %12, %%pmc12\n\t"
        "smir %13, %%pmc13\n\t"
        "smir %14, %%pmc14\n\t"
        "smir %15, %%pmc15\n\t"
        :
        "=r"(hwc[0]),
        "=r"(hwc[1]),
        "=r"(hwc[2]),
        "=r"(hwc[3]),
        "=r"(hwc[4]),
        "=r"(hwc[5]),
        "=r"(hwc[6]),
        "=r"(hwc[7]),
        "=r"(hwc[8]),
        "=r"(hwc[9]),
        "=r"(hwc[10]),
        "=r"(hwc[11]),
        "=r"(hwc[12]),
        "=r"(hwc[13]),
        "=r"(hwc[14]),
        "=r"(hwc[15])
    );
}

long long *vftr_get_all_ve_counters () {
   long long *counters = (long long*)malloc(VE_MAX_HWC_EVENTS * sizeof(long long));
   vftr_read_sxhwc_registers (counters);
   counters[0] &= 0x000fffffffffffff; /* 52bit counter */
   counters[1] &= 0x000fffffffffffff; /* 52bit counter */
   for (int i = 2; i < VE_MAX_HWC_EVENTS; i++) {
      counters[i] &= 0x00ffffffffffffff; /* 56bit counter */
   }
   return counters;
}

long long *vftr_get_active_ve_counters () {
   long long *all_counters = vftr_get_all_ve_counters();
   int n_active = vftrace.hwprof_state.n_counters;
   long long *active_counters = (long long *)malloc(vftrace.hwprof_state.n_counters * sizeof(long long));  
   int idx = 0;
   for (int i = 0; i < VE_MAX_HWC_EVENTS; i++) {
      if (vftrace.hwprof_state.veprof.active_counters[i]) active_counters[idx++] = all_counters[i];
   } 
   free(all_counters); 
   return active_counters;
}

