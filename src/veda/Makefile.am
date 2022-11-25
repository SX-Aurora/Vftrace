noinst_LTLIBRARIES = libvftr_veda.la

libvftr_veda_la_SOURCES =

libvftr_veda_la_SOURCES += veda_regions.c
libvftr_veda_la_SOURCES += vedaprofiling.c \
			   collate_vedaprofiles.c \
			   logfile_veda_table.c
libvftr_veda_la_SOURCES += veda_callback.c \
			   veda_callback_launch_kernel.c

libvftr_veda_la_CFLAGS = -I${srcdir}/../ \
			 -I${srcdir}/../sorting/ \
			 -D_VEDA

if SELF_PROFILING
libvftr_veda_la_CFLAGS += -D_SELF_PROFILE_VFTRACE
endif
