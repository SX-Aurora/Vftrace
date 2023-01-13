#include <stdio.h>

#include "vftrace_state.h"

#include "calculator.h"

int main (int argc, char *argv[]) {
   // Calculator gets two formulas: a + b and a - b.
   // First, we set a = 1 and b = 1 and check that a + b = 2 and a - b = 0.
   // Second, we set a = -1 and b = 1 and check that a + b = 0 and a - b = -2;

   int n_variables = 2; // a and b
   int n_observables = 2; // sum and diff

   char *symbols[] = {"a", "b"};
   char *formulas[] = {"a + b", "a - b"};

   
   vftrace.hwprof_state.n_counters = n_variables;
   vftr_calculator_t calc = vftr_init_calculator (n_observables, symbols, formulas);

   long long values1[2] = {1, 1};
   double sum, diff;
   vftr_set_calculator_counters (&calc, values1);  
   sum = vftr_calculator_evaluate (calc, 0);
   diff = vftr_calculator_evaluate (calc, 1);

   printf ("1 + 1 = %.0f\n", sum);
   printf ("1 - 1 = %.0f\n", diff);

   long long values2[2] = {-1, 1}; 
   vftr_set_calculator_counters (&calc, values2);  
   sum = vftr_calculator_evaluate (calc, 0);
   diff = vftr_calculator_evaluate (calc, 1);

   printf ("-1 + 1 = %.0f\n", sum);
   printf ("-1 - 1 = %.0f\n", diff);

   return 0; 
}
