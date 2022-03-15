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

   void *value_arrays[9] = {
      (void*) char_arr,
      (void*) string_arr,
      (void*) int_arr,
      (void*) long_arr,
      (void*) longlong_arr,
      (void*) float_arr,
      (void*) double_arr,
      (void*) longdouble_arr,
      (void*) bool_arr};

   // print horizontal lines
   bool hlines[4] = {true, true, false, true};
   bool vlines[3] = {true, true, true};
   bool novlines[3] = {false, false, false};
   column_t types[9] = {col_char, col_string, col_int, col_long, col_longlong, col_float, col_double, col_longdouble, col_bool};
   char *headers[9] = {"chars", "strings", "integers", "long hex integers", "ll integers", "floats", "doubles", "long doubles", "bools"};
   char *emptyheaders[9] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
   char *formats[9] = {"char = %c", "%s", "%4d", "0x%06lx", "%6lld", "%3.1f", "%6.2lf", "%6.3llf", "val = %s"};
   char lalign[9] = {'l', 'l','l','l','l','l','l','l','l'};
   char *hlalign = lalign;
   char calign[9] = {'c', 'c','c','c','c','c','c','c','c'};
   char *hcalign = calign;
   char ralign[9] = {'r', 'r','r','r','r','r','r','r','r'};
   char *hralign = ralign;

   vftr_print_table(stdout,
                    9,
                    5,
                    hlines, vlines,
                    types,
                    headers,
                    formats,
                    hcalign,
                    lalign,
                    value_arrays);

   fprintf(stdout, "\n");

   vftr_print_table(stdout,
                    9,
                    5,
                    hlines, vlines,
                    types,
                    headers,
                    formats,
                    hlalign,
                    calign,
                    value_arrays);

   fprintf(stdout, "\n");

   vftr_print_table(stdout,
                    9,
                    5,
                    hlines, vlines,
                    types,
                    headers,
                    formats,
                    hlalign,
                    ralign,
                    value_arrays);

   fprintf(stdout, "\n");

   vftr_print_table(stdout,
                    9,
                    5,
                    hlines, novlines,
                    types,
                    emptyheaders,
                    formats,
                    hlalign,
                    ralign,
                    value_arrays);
   return 0;
}