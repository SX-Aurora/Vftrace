#include "likwid.h"

#include "vftrace_state.h"
#include "signal_handling.h"

enum {POWER_START, POWER_STOP};

int vftr_likwid_start_stop_power (likwid_state_t *l, int mode) {
   int err = 0;
   for (int s = 0; s < l->n_sockets; s++) {
      int cpu_id = l->socket_cores[s];
      for (int pdom = 0; pdom < NUM_POWER_DOMAINS; pdom++) {
         if (l->pdom_active[pdom]) {
            if (mode == POWER_START) {
               err |= power_start (l->pd[s][pdom], cpu_id, pdom);
            } else {
               err |= power_stop (l->pd[s][pdom], cpu_id, pdom);
            }
         }
      }
   }
   return err;
}

double vftr_likwid_energy_for_all_sockets (hwprof_state_t state, int pdom) {
   double muj_tot = 0;
   for (int s = 0; s < state.likwid.n_sockets; s++) {
      muj_tot += power_printEnergy (state.likwid.pd[s][pdom]) * 1e6;
   }
   return muj_tot;
}

void vftr_likwid_init (hwprof_state_t *state) {
   printf ("INIT LIKWID!\n");
   if (topology_init() < 0) {
      fprintf (stderr, "Vftrace Error: Likwid: topology_init failed.\n");
      vftr_abort(0);
   }

   state->likwid.tp = get_cpuTopology(); 

   if (power_init(0) < 0) {
      fprintf (stderr, "Vftrace Error: Likwid: topology_init failed.\n");
      vftr_abort(0);
   }

   AffinityDomains_t aff = get_affinityDomains(); 
   int n_aff_domains = aff->numberOfAffinityDomains;
   state->likwid.n_sockets = 0;
   for (int i = 0; i < n_aff_domains; i++) {
      AffinityDomain domain = aff->domains[i];
      int sock_id;
      if ((sscanf (domain.tag, "S%d", &sock_id) == 1)) {
         state->likwid.n_sockets++;
      }

   }
   state->likwid.socket_cores = (int*)malloc (state->likwid.n_sockets * sizeof(int));
   int j = 0;
   for (int i = 0; i < n_aff_domains; i++) {
      AffinityDomain domain = aff->domains[i];
      int sock_id;
      if ((sscanf (domain.tag, "S%d", &sock_id) == 1)) {
         int core = domain.processorList[0];
         HPMaddThread(core);       
         state->likwid.socket_cores[j++] = core;
      }
   }

   for (int i = 0; i < NUM_POWER_DOMAINS; i++) {
      state->likwid.pdom_active[i] = false;
   }
   for (int i = 0; i < state->n_counters; i++) {
      char *name = state->counters[i].name;
      if (!strcmp(name, "PKG")) {
         state->likwid.pdom_active[PKG] = true;
      } else if (!strcmp(name, "PP0")) {
         state->likwid.pdom_active[PP0] = true;
      } else if (!strcmp(name, "PP1")) {
         state->likwid.pdom_active[PP1] = true;
      } else if (!strcmp(name, "DRAM")) {
         state->likwid.pdom_active[DRAM] = true;
      } else if (!strcmp(name, "PLATFORM")) {
         state->likwid.pdom_active[PLATFORM] = true;
      }
   }
   
   state->likwid.pd = (PowerData_t**)malloc(state->likwid.n_sockets * sizeof(PowerData_t*));
   for (int i = 0; i < NUM_POWER_DOMAINS; i++) {
      state->likwid.pd[i] = (PowerData_t*)malloc(NUM_POWER_DOMAINS * sizeof(PowerData_t));
   }
   (void)vftr_likwid_start_stop_power (&(state->likwid), POWER_START);
   state->likwid.total_energy = 0;
}

void vftr_likwid_finalize () {
   ///power_stop (vftrace.hwprof_state.likwid.pd, 0, PKG); 
   vftr_likwid_start_stop_power (&(vftrace.hwprof_state.likwid), POWER_STOP);
   for (int s = 0; s < vftrace.hwprof_state.likwid.n_sockets; s++) {
      free(vftrace.hwprof_state.likwid.pd[s]);
   }
   free (vftrace.hwprof_state.likwid.pd);
   power_finalize ();
   topology_finalize ();
}

long long *vftr_get_likwid_counters () {
   
   (void)vftr_likwid_start_stop_power (&(vftrace.hwprof_state.likwid), POWER_STOP);

   int n = vftrace.hwprof_state.n_counters;
   long long *counters  = (long long*)malloc(n * sizeof(long long));

   ///double muj = power_printEnergy (vftrace.hwprof_state.likwid.pd) * 1e6;
   ///vftrace.hwprof_state.likwid.total_energy += muj;
   int c = 0;
   for (int pdom = 0; pdom < NUM_POWER_DOMAINS; pdom++) {
      if (vftrace.hwprof_state.likwid.pdom_active[pdom]) {
         counters[c++] = (long long)vftr_likwid_energy_for_all_sockets (vftrace.hwprof_state, pdom);
      }
   }

   printf ("COUNTERS: %d\n", counters[0]);
  
   ///power_start (vftrace.hwprof_state.likwid.pd, 0, PKG);
   (void)vftr_likwid_start_stop_power (&(vftrace.hwprof_state.likwid), POWER_START);
   printf ("GET COUNTERS 2: %d %d\n", vftrace.hwprof_state.likwid.socket_cores[0],
           vftrace.hwprof_state.likwid.socket_cores[1]);
   return counters;
}
