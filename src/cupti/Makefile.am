noinst_LTLIBRARIES = libvftr_cupti.la

libvftr_cupti_la_SOURCES = 

libvftr_cupti_la_SOURCES += cupti_init_final.c \
                            callbacks.c \
                            cupti_event_list.c \
                            cuptiprofiling.c \
                            collate_cuptiprofiles.c \
			    gpu_info.c \
                            logfile_cupti_table.c \
                            ranklogfile_cupti_table.c

libvftr_cupti_la_CFLAGS = -I${srcdir}/../ -D_CUPTI

#if HAS_LIBERTY
libvftr_cupti_la_CFLAGS += -D_LIBERTY
#endif
