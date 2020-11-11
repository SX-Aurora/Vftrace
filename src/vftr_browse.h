#ifndef VFTR_BROWSE_H
#define VFTR_BROWSE_H

#include "vftr_filewrite.h"
#include "vftr_stacks.h"

enum origin_page {HOME, TREE, PROFILE};

void vftr_browse_make_html_indent (FILE *fp, int n_indent_0, int n_indent_extra);
void vftr_browse_print_css_header (FILE *fp);
void vftr_browse_print_stacktree_page (FILE *fp_out, bool is_empty, char *func_names[], int n_funcs, int this_i_func,
			             stack_leaf_t *leaf, double *imbalances, double total_time);
void vftr_browse_print_navigation_bars (FILE *fp, char *func_names[], int n_funcs, int this_i_func, enum origin_page op);
void vftr_browse_print_index_html (char *func_names[], int n_funcs);

void vftr_browse_create_directory ();
FILE *vftr_browse_init_profile_table ();
void vftr_browse_create_profile_header (FILE *fp);
void vftr_browse_print_table_line (FILE *fp, int stack_id, int n_calls,
				 int format_excl, int format_incl, double t_excl, double t_incl,
				 double t_rel, double t_cum, char *func_name, char *call_name);
void vftr_browse_finalize_table (FILE *fp);

int vftr_browse_test_1 (FILE *fp_in, FILE *fp_out);
#endif
