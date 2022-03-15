#ifndef TABLE_TYPES_H
#define TABLE_TYPES_H

#include <stdbool.h>

typedef enum {
   col_none,
   col_char,
   col_string,
   col_int,
   col_long,
   col_longlong,
   col_float,
   col_double,
   col_longdouble,
   col_bool
} column_kind;

typedef struct {
   column_kind kind;
   char *header;
   char *format;
   char headeralign;
   char align;
   void *values;
} column_t;

typedef struct {
   int ncols;
   int nrows;
   bool hlines[4];
   bool vlines[4];
   column_t *columns;
} table_t;

#endif
