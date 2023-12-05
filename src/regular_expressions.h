#ifndef REGULAR_EXPRESSIONS_H
#define REGULAR_EXPRESSIONS_H

#include <stdbool.h>
#include <regex.h>

regex_t *vftr_compile_regexp(char *s);

bool vftr_pattern_match(regex_t *r, char *s);

void vftr_free_regexp (regex_t *r);

#endif
