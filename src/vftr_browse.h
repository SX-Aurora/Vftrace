#ifndef VFTR_BROWSE_H
#define VFTR_BROWSE_H

#include "vftr_functions.h"
#include "vftr_filewrite.h"
#include "vftr_stacks.h"

enum origin_page {HOME, TREE, PROFILE};

void vftr_browse_make_html_indent (FILE *fp, int n_indent_0, int n_indent_extra);
void vftr_browse_print_css_header (FILE *fp, int n_chars_max, int n_final);
void vftr_browse_create_stat_directories (display_function_t **display_functions, int i_func);
void vftr_browse_print_stacktree_page (FILE *fp_out, bool is_empty, display_function_t **display_functions, int i_func, int n_funcs,
			               stack_leaf_t *leaf, double *imbalances, double total_time,
				       int n_chars_max, int n_final);
void vftr_browse_print_navigation_bars (FILE *fp, display_function_t **display_functions, int i_func, int n_funcs, enum origin_page op);
void vftr_browse_print_index_html (display_function_t **display_functions, int n_funcs);

void vftr_browse_create_directory ();
FILE *vftr_browse_init_profile_table (display_function_t **display_functions, int n_display_functions);
void vftr_browse_create_profile_header (FILE *fp);
void vftr_browse_print_table_line (FILE *fp, int stack_id,
				   long long application_runtime_usec, double sampling_overhead_time,
				   int n_calls, long long t_excl_usec, long long t_incl_usec, long long t_sum_usec, double t_overhead,
				   char *func_name, char *caller_name, column_t *prof_columns, bool mark_disp_f);
void vftr_browse_finalize_table (FILE *fp);

int vftr_browse_test_1 (FILE *fp_in, FILE *fp_out);
#endif
