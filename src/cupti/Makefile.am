noinst_LTLIBRARIES = libvftr_cupti.la

libvftr_cupti_la_SOURCES = 

libvftr_cupti_la_SOURCES += cupti_init_final.c \
                            callbacks.c \
                            cuptiprofiling.c \
                            collate_cuptiprofiles.c \
                            cupti_logfile.c \
                            cupti_ranklogfile.c \
                            cupti_utils.c

libvftr_cupti_la_CFLAGS = -I${srcdir}/../ -D_CUPTI

#if HAS_LIBERTY
libvftr_cupti_la_CFLAGS += -D_LIBERTY
#endif
