#ifndef CONFIGURATION_PRINT_H
#define CONFIGURATION_PRINT_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "configuration_types.h"

void vftr_print_config(FILE *fp, config_t config, bool show_title);

#endif
