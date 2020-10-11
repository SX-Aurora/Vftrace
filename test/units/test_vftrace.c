#include <stdio.h>
#include <string.h>

#include "vftr_setup.h"
#include "vftr_filewrite.h"
#include "vftr_functions.h"
#include "vftr_scenarios.h"
#include "vftr_stacks.h"
#include "vftr_symbols.h"
#include "vftr_environment.h"
#include "vftr_hwcounters.h"
#include "vftr_html.h"

int this_fails () {
	return 1;
}

int this_passes () {
	return 0;
}

#define OUTFILE_NAME_BUF 50

int main (int argc, char **argv) {

#if defined(_MPI)
	PMPI_Init (NULL, NULL);
	vftr_get_mpi_info (&vftr_mpirank, &vftr_mpisize);
#else
	vftr_mpirank = 0;
	vftr_mpisize = 1;
#endif
	
	if (argc < 2) {
		printf ("Usage: test_vftrace <test_name> [<input_file>]\n");
		return -1;
	}
    	int retval;
	char outfilename[OUTFILE_NAME_BUF];
	if (vftr_mpisize > 1) {
		snprintf (outfilename, OUTFILE_NAME_BUF, "%s.out_%d", argv[1], vftr_mpirank);
	} else {
		snprintf (outfilename, OUTFILE_NAME_BUF, "%s.out", argv[1]);
	}
	FILE *fp_out = fopen (outfilename, "w+");

	FILE *fp_in = NULL;
	if (argc > 2) {
		fp_in = fopen (argv[2], "r");
	}
	if (!strcmp (argv[1], "this_fails")) {
		retval = this_fails();
	} else if (!strcmp (argv[1], "this_passes")) {
		retval = this_passes();
	} else if (!strcmp (argv[1], "vftr_symbols_test_1")) {
		retval = vftr_symbols_test_1 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_environment_test_1")) {
		retval = vftr_environment_test_1 (fp_out);
	} else if (!strcmp (argv[1], "vftr_setup_test_1")) {
		retval = vftr_setup_test_1 (fp_out);
	} else if (!strcmp (argv[1], "vftr_setup_test_2")) {
		retval = vftr_setup_test_2 (fp_out);
	} else if (!strcmp (argv[1], "vftr_filewrite_test_1")) {
		retval = vftr_filewrite_test_1 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_filewrite_test_2")) {
		retval = vftr_filewrite_test_2 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_scenario_test_1")) {
		retval = vftr_scenario_test_1 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_scenario_test_2")) {
		retval = vftr_scenario_test_2 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_scenario_test_3")) {
		retval = vftr_scenario_test_3 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_functions_test_1")) {
		retval = vftr_functions_test_1 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_functions_test_2")) {
		retval = vftr_functions_test_2 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_functions_test_3")) {
		retval = vftr_functions_test_3 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_functions_test_4")) {
		retval = vftr_functions_test_4 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_functions_test_5")) {
		retval = vftr_functions_test_5 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_stacks_test_1")) {
		retval = vftr_stacks_test_1 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_stacks_test_2")) {
		retval = vftr_stacks_test_2 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_veperf_test_1")) {
		retval = vftr_veperf_test_1 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_veperf_test_2")) {
		retval = vftr_veperf_test_2 (fp_in, fp_out);
	} else if (!strcmp (argv[1], "vftr_html_test_1")) {
		retval = vftr_html_test_1 (fp_in, fp_out);
	} else {
		printf ("No matching test found\n");
	}
	fclose (fp_out);
	fp_out = NULL;
	if (fp_in) fclose (fp_in);
#ifdef _MPI
        PMPI_Finalize ();
#endif
	return retval;
}
