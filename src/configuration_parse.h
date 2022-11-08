#ifndef CONFIGURATION_PARSE_H
#define CONFIGURATION_PARSE_H

#include "configuration_types.h"

int vftr_parse_config_check_json_format(char *config_string);

void vftr_parse_config(char *config_string, config_t *config_ptr);

#endif
