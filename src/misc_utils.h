#ifndef MISC_UTILS_H
#define MISC_UTILS_H

#include <stdbool.h>

int vftr_count_base_digits(long long value, int base);

char *vftr_bool_to_string(bool value);

int vftr_levenshtein_distance(char *a, char *b);

void vftr_chop_trailing_char(char *string, char trailing_char);

#endif
