#include "likwid.h"

#include "vftrace_state.h"

void vftr_likwid_init (hwprof_state_t *state) {
   printf ("INIT LIKWID!\n");
   if (topology_init() < 0) {
      fprintf (stderr, "Vftrace Error: Likwid: topology_init failed.\n");
      vftr_abort(0);
   }

   if (power_init(0) < 0) {
      fprintf (stderr, "Vftrace Error: Likwid: topology_init failed.\n");
      vftr_abort(0);
   }
   
   printf ("MALLOC STATE\n");
   state->likwid.pd = (PowerData_t)malloc(sizeof(PowerData_t));
   printf ("START POWER!\n");
   power_start (state->likwid.pd, 0, PKG);
   printf ("LIKWID INITIALIZED!\n");
   state->likwid.total_energy = 0;
}

void vftr_likwid_finalize () {
   power_stop (vftrace.hwprof_state.likwid.pd, 0, PKG); 
   free (vftrace.hwprof_state.likwid.pd);
   power_finalize ();
   topology_finalize ();
}

long long *vftr_get_likwid_counters () {
   
   if (power_stop (vftrace.hwprof_state.likwid.pd, 0, PKG) < 0) {
       printf ("Failed to stop power!\n");
   }
   int n = vftrace.hwprof_state.n_counters;
   long long *counters  = (long long*)malloc(n * sizeof(long long));
   double muj = power_printEnergy (vftrace.hwprof_state.likwid.pd) * 1e6;
   vftrace.hwprof_state.likwid.total_energy += muj;
   counters[0] = (long long)vftrace.hwprof_state.likwid.total_energy;
   printf ("RETURN: %lld\n", counters[0]);
   power_start (vftrace.hwprof_state.likwid.pd, 0, PKG);
   return counters;
}
