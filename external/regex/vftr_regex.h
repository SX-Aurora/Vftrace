#ifndef VFTR_REGEX_H
#define VFTR_REGEX_H

#include "regex.h"

int vftr_regcomp (regex_t *preg, const char *pattern, int cflags);
int vftr_regexec (regex_t *preg, const char *string, size_t nmatch, regmatch_t pmatch[], int eflags);

#endif
