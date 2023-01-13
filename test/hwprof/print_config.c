#include "configuration.h"
#include "configuration_print.h"

int main (int argc, char *argv[]) {
  config_t config = vftr_read_config(); 

  vftr_print_config (stdout, config, true);
}
