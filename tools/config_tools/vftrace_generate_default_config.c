#include <stdio.h>

#include "configuration_types.h"
#include "configuration_defaults.h"
#include "configuration_print.h"

int main(int argc, char **argv) {

   config_t config = vftr_set_config_default();
   vftr_print_config(stdout, config, false);
}
