#include "vftr_hwcounters.h"
#include "vftr_setup.h"
#include "vftr_environment.h"
#ifdef _MPI
#include <mpi.h>
#endif

int main (int argc, char **argv) {

#if defined(_MPI)
  PMPI_Init(&argc, &argv);
  vftr_get_mpi_info (&vftr_mpirank, &vftr_mpisize);
#else 
  vftr_mpirank = 0;
  vftr_mpisize = 1;
#endif

  vftr_read_environment();

#if defined(HAS_SXHWC)
#define N_DIGITS 6
  const char *sx_counter_names[16] = {"EX", "VX", "FPEC", "VE", "VECC", "L1MCC", 
  	"VE2", "VAREC", "VLDEC", "PCCC", "VLPC", "VLEC", "VLCME", "FMAEC", "PTCC", "TTCC"};
  int n = 100000;
  int n_iter = 1000;
  double x[n], y[n], z[n];
  long long *c1, *c2;
  long long c_diff[MAX_HWC_EVENTS][n_iter];
  c1 = (long long *)malloc (16 * sizeof(long long));
  c2 = (long long *)malloc (16 * sizeof(long long));
  fprintf (stdout, "Checking reproducibility of SX Aurora hardware counters\n");
  fprintf (stdout, "Averaging over %d iterations\n", n_iter);
  
  for (int i = 0; i < n; i++) {
  	x[i] = i;
  	y[i] = 0.5 * i;
  }
  for (int n = 0; n < n_iter; n++) {
  	vftr_read_sxhwc_registers (c1);
  	for (int i = 0; i < n; i++) {
  		z[i] = x[i] + x[i] * y[i];
  	}
  	vftr_read_sxhwc_registers (c2);
  	for (int i = 0; i < 16; i++) {
  		c_diff[i][n] = c2[i] - c1[i];
  	}
  }
  
  double c_avg[MAX_HWC_EVENTS];
  long long sum_c;
  for (int i = 0; i < 16; i++) {
  	sum_c = 0;
  	for (int n = 0; n < n_iter; n++) {
  		sum_c += c_diff[i][n];
  	}
  	c_avg[i] = (double)sum_c / n_iter;
  }
  // The counters ending with a "C" are clock counters. They depend on the momentary performance of
  // the system. Therefore, the mean value is not reliable and is therefore not printed. We constrain
  // this output only to the hardware counters without a "C".
  fprintf (stdout, "%*s: %*d\n", N_DIGITS, sx_counter_names[0], N_DIGITS, (int)floor(c_avg[0])); // EX
  fprintf (stdout, "%*s: %*d\n", N_DIGITS, sx_counter_names[1], N_DIGITS, (int)floor(c_avg[1])); // VX 
  fprintf (stdout, "%*s: %*d\n", N_DIGITS, sx_counter_names[3], N_DIGITS, (int)floor(c_avg[3])); // VE
  fprintf (stdout, "%*s: %*d\n", N_DIGITS, sx_counter_names[6], N_DIGITS, (int)floor(c_avg[6])); // VE2
  fprintf (stdout, "%*s: %*d\n", N_DIGITS, sx_counter_names[12], N_DIGITS, (int)floor(c_avg[12])); // VLCME
  free(c1);
  free(c2);
#endif


#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}

