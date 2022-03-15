#include <stdio.h>

#include "tables.h"

int main() {
   // column data arrays
   char char_arr[5] = {'a', 'b', 'c', 'd', 'e'};
   char *string_arr[5] = {"string", "slightly longer string", "str", "my string", "last string"};
   int int_arr[5] = {1, 2, 3, 4, 5};
   long long_arr[5] = {101,102,103,104,105};
   long long longlong_arr[5] = {1001, 1002, 1003, 1004, 1005};
   float float_arr[5] = {1.0,1.1,1.2,1.3,1.4};
   double double_arr[5] = {137.5, 3.1415, 2.7182818, 1.0/3.0, 1.1412};
   long double longdouble_arr[5] = {10.4, 11.4, 12.4, 13.4, 14.4};
   bool bool_arr[5] = {true, false, true, true, false};

   char *headers[9] = {"chars", "strings", "integers", "long hex integers", "ll integers", "floats", "doubles", "long doubles", "bools"};
   char *formats[9] = {"char = %c", "%s", "%4d", "0x%06lx", "%6lld", "%3.1f", "%6.2lf", "%6.3llf", "val = %s"};
   column_kind kinds[9] = {col_char, col_string, col_int, col_long, col_longlong, col_float, col_double, col_longdouble, col_bool};

   // first table
   table_t table = vftr_new_table();
   vftr_table_set_nrows(&table, 5);
   vftr_table_add_column(&table, kinds[0], headers[0], formats[0], 'c', 'l', (void*) char_arr);
   vftr_table_add_column(&table, kinds[1], headers[1], formats[1], 'c', 'l', (void*) string_arr);
   vftr_table_add_column(&table, kinds[2], headers[2], formats[2], 'c', 'l', (void*) int_arr);
   vftr_table_add_column(&table, kinds[3], headers[3], formats[3], 'c', 'l', (void*) long_arr);
   vftr_table_add_column(&table, kinds[4], headers[4], formats[4], 'c', 'l', (void*) longlong_arr);
   vftr_table_add_column(&table, kinds[5], headers[5], formats[5], 'c', 'l', (void*) float_arr);
   vftr_table_add_column(&table, kinds[6], headers[6], formats[6], 'c', 'l', (void*) double_arr);
   vftr_table_add_column(&table, kinds[7], headers[7], formats[7], 'c', 'l', (void*) longdouble_arr);
   vftr_table_add_column(&table, kinds[8], headers[8], formats[8], 'c', 'l', (void*) bool_arr);

   vftr_print_table(stdout, table);
   vftr_table_free(&table);

   fprintf(stdout, "\n");

   vftr_table_set_nrows(&table, 5);
   vftr_table_add_column(&table, kinds[0], headers[0], formats[0], 'l', 'c', (void*) char_arr);
   vftr_table_add_column(&table, kinds[1], headers[1], formats[1], 'l', 'c', (void*) string_arr);
   vftr_table_add_column(&table, kinds[2], headers[2], formats[2], 'l', 'c', (void*) int_arr);
   vftr_table_add_column(&table, kinds[3], headers[3], formats[3], 'l', 'c', (void*) long_arr);
   vftr_table_add_column(&table, kinds[4], headers[4], formats[4], 'l', 'c', (void*) longlong_arr);
   vftr_table_add_column(&table, kinds[5], headers[5], formats[5], 'l', 'c', (void*) float_arr);
   vftr_table_add_column(&table, kinds[6], headers[6], formats[6], 'l', 'c', (void*) double_arr);
   vftr_table_add_column(&table, kinds[7], headers[7], formats[7], 'l', 'c', (void*) longdouble_arr);
   vftr_table_add_column(&table, kinds[8], headers[8], formats[8], 'l', 'c', (void*) bool_arr);

   vftr_print_table(stdout, table);
   vftr_table_free(&table);

   fprintf(stdout, "\n");

   vftr_table_set_nrows(&table, 5);
   vftr_table_add_column(&table, kinds[0], headers[0], formats[0], 'l', 'r', (void*) char_arr);
   vftr_table_add_column(&table, kinds[1], headers[1], formats[1], 'l', 'r', (void*) string_arr);
   vftr_table_add_column(&table, kinds[2], headers[2], formats[2], 'l', 'r', (void*) int_arr);
   vftr_table_add_column(&table, kinds[3], headers[3], formats[3], 'l', 'r', (void*) long_arr);
   vftr_table_add_column(&table, kinds[4], headers[4], formats[4], 'l', 'r', (void*) longlong_arr);
   vftr_table_add_column(&table, kinds[5], headers[5], formats[5], 'l', 'r', (void*) float_arr);
   vftr_table_add_column(&table, kinds[6], headers[6], formats[6], 'l', 'r', (void*) double_arr);
   vftr_table_add_column(&table, kinds[7], headers[7], formats[7], 'l', 'r', (void*) longdouble_arr);
   vftr_table_add_column(&table, kinds[8], headers[8], formats[8], 'l', 'r', (void*) bool_arr);

   vftr_print_table(stdout, table);
   vftr_table_free(&table);

   fprintf(stdout, "\n");

   vftr_table_set_nrows(&table, 5);
   vftr_table_add_column(&table, kinds[0], NULL, formats[0], 'l', 'r', (void*) char_arr);
   vftr_table_add_column(&table, kinds[1], NULL, formats[1], 'l', 'r', (void*) string_arr);
   vftr_table_add_column(&table, kinds[2], NULL, formats[2], 'l', 'r', (void*) int_arr);
   vftr_table_add_column(&table, kinds[3], NULL, formats[3], 'l', 'r', (void*) long_arr);
   vftr_table_add_column(&table, kinds[4], NULL, formats[4], 'l', 'r', (void*) longlong_arr);
   vftr_table_add_column(&table, kinds[5], NULL, formats[5], 'l', 'r', (void*) float_arr);
   vftr_table_add_column(&table, kinds[6], NULL, formats[6], 'l', 'r', (void*) double_arr);
   vftr_table_add_column(&table, kinds[7], NULL, formats[7], 'l', 'r', (void*) longdouble_arr);
   vftr_table_add_column(&table, kinds[8], NULL, formats[8], 'l', 'r', (void*) bool_arr);

   vftr_table_left_outline(&table, false);
   vftr_table_right_outline(&table, false);
   vftr_table_columns_separating_line(&table, false);

   vftr_print_table(stdout, table);
   vftr_table_free(&table);

   return 0;
}
