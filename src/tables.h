#ifndef TABLES_H
#define TABLES_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

typedef enum {
   col_char,
   col_string,
   col_int,
   col_long,
   col_longlong,
   col_float,
   col_double,
   col_longdouble,
   col_bool
} column_t;

void vftr_print_table(FILE *fp, int ncols, int nrows, bool rules[3],
                      column_t *coltypes, char **headers,
                      char **formats, char *align, void **value_lists);

#endif
