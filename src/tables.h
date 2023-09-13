#ifndef TABLES_H
#define TABLES_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "table_types.h"

void vftr_print_table(FILE *fp, table_t table);

table_t vftr_new_table();

void vftr_table_free(table_t *table_ptr);

void vftr_table_set_nrows(table_t *table_ptr, int nrows);

void vftr_table_add_column(table_t *table_ptr, column_kind kind,
                           char *header, char *format,
                           char headeralign, char align,
                           void *values);

void vftr_table_left_outline(table_t *table, bool value);

void vftr_table_right_outline(table_t *table, bool value);

void vftr_table_columns_separating_line(table_t *table, bool value);

void vftr_table_top_outline(table_t *table, bool value);

void vftr_table_bottom_outline(table_t *table, bool value);

void vftr_table_header_separating_line(table_t *table, bool value);

void vftr_table_rows_separating_line(table_t *table, bool value);

void vftr_print_table_hline (FILE *fp, int ncols, bool vlines[3], int *widths);

#endif
