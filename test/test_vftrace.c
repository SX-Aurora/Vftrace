#include <stdio.h>
#include <string.h>

#include "vftr_symbols.h"

int this_fails () {
	return 1;
}

int this_passes () {
	return 0;
}

//int vftr_symbols_test_1 (FILE *fp) {
//	printf ("Get symtab\n");
//	//vftr_get_library_symtab ("", fp, 0L, 0);	
//	return 0;
//}
//

int main (int argc, char **argv) {
	
	if (argc < 2) {
		printf ("Usage: test_vftrace <test_name> [<input_file>]\n");
		return -1;
	}
    	int retval;

        char *basename = strdup (argv[1]);
	char *outfilename = strcat (basename, ".out");
	FILE *fp_out = fopen (outfilename, "w");

	FILE *fp_in;
	if (argc > 2) {
		fp_in = fopen (argv[2], "r");
	}
	//FILE *fp_in = fopen ("test.x", "r");
	if (!strcmp (argv[1], "this_fails")) {
		retval = this_fails();
	} else if (!strcmp (argv[1], "this_passes")) {
		retval = this_passes();
	} else if (!strcmp (argv[1], "vftr_symbols_test_1")) {
		retval = vftr_symbols_test_1 (fp_in, fp_out);
	} else {
		printf ("No matching test found\n");
	}
	fclose (fp_out);
	if (fp_in) fclose (fp_in);
	printf ("FOO\n");
}
