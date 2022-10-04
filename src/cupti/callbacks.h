#ifndef CALLBACKS_H
#define CALLBACKS_H

#include <cupti.h>

void CUPTIAPI cupti_event_callback (void *userdata,
                                    CUpti_CallbackDomain domain,
                                    CUpti_CallbackId cbid,
                                    const CUpti_CallbackData *cb_info);
#endif
