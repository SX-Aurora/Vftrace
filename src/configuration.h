#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <stdlib.h>

#include "configuration_types.h"

bool vftr_read_environment_vftr_off();

char *vftr_read_environment_vftr_config();

config_t vftr_read_config();

void vftr_config_free(config_t *config_ptr);

#endif
