#include "vftr_scenarios.h"

int main (int argc, char **argv) {

#ifdef _MPI
  PMPI_Init(&argc, &argv);
#endif

  if (argc != 2) {
    printf ("scenarios_1: Missing input file argument.\n");
    return 1;
  }

  FILE *fp_in = fopen (argv[1], "r");
  if (fp_in == NULL) {
    printf ("Scenario file %s not found!\n", argv[1]);
    return 1;
  } 

  vftr_read_scenario_file ("", fp_in);
  fprintf (stdout, "Registered variables: %d\n", vftr_scenario_expr_n_vars);
  for (int i = 0; i < vftr_scenario_expr_n_vars; i++) {
  	fprintf (stdout, "%d: name: %s\n", i, te_vars[i].name);		
  }
  fprintf (stdout, "Check for the three additional entries: \n");
  fprintf (stdout, "%s\n", te_vars[vftr_scenario_expr_n_vars].name);
  fprintf (stdout, "%s\n", te_vars[vftr_scenario_expr_n_vars+1].name);
  fprintf (stdout, "%s\n", te_vars[vftr_scenario_expr_n_vars+2].name);
  fprintf (stdout, "Registered formulas: %d\n", vftr_scenario_expr_n_formulas);
  for (int i = 0; i < vftr_scenario_expr_n_formulas; i++) {
  	vftr_scenario_print_formula (stdout, vftr_scenario_expr_formulas[i]);
  }

#ifdef _MPI
  PMPI_Finalize();
#endif

  return 0;
}

