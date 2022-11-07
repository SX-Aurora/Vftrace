#ifndef MISC_UTILS_H
#define MISC_UTILS_H

int vftr_count_base_digits(long long value, int base);

char *vftr_read_file_to_string(char *filename);

int vftr_levenshtein_distance(char *a, char *b);

void vftr_chop_trailing_char(char *string, char trailing_char);

void vftr_trim_left_with_delimiter(char *string, char *delim);

char *vftr_combine_string_and_address(const char *str, const void *addr);

char *vftr_byte_unit(unsigned long long size);

#endif
