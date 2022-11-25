#include <stdlib.h>

#include <veda.h>

#include "veda_regions.h"

void vftr_veda_callback_launch_kernel_enter(VEDAprofiler_data* data) {
   vftr_veda_region_begin("vedaLaunchKernel");
   VEDAprofiler_vedaLaunchKernel *LaunchKernelData;
   LaunchKernelData = (VEDAprofiler_vedaLaunchKernel*) &(data->type);
   vftr_veda_region_begin(LaunchKernelData->kernel);
}

void vftr_veda_callback_launch_kernel_exit(VEDAprofiler_data* data) {
   VEDAprofiler_vedaLaunchKernel *LaunchKernelData;
   LaunchKernelData = (VEDAprofiler_vedaLaunchKernel*) &(data->type);
   vftr_veda_region_end(LaunchKernelData->kernel);
   vftr_veda_region_end("vedaLaunchKernel");
}
