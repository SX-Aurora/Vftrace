#ifndef VFTR_HTML_H
#define VFTR_HTML_H

#include "vftr_stacks.h"

void vftr_print_html_output (FILE *fp_out, char *func_name, stack_leaf_t *leaf, double *imbalances);

int vftr_html_test_1 (FILE *fp_in, FILE *fp_out);
#endif
