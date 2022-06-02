noinst_LTLIBRARIES = libvftr_omp.la

libvftr_omp_la_SOURCES =

libvftr_omp_la_SOURCES += omp_regions.c

libvftr_omp_la_SOURCES += start_tool.c \
                          initialize.c \
                          finalize.c

libvftr_omp_la_SOURCES += parallel_begin.c \
                          parallel_end.c

libvftr_omp_la_SOURCES += implicit_task.c

libvftr_omp_la_CFLAGS =
libvftr_omp_la_CFLAGS += -I${srcdir}/../ -D_OMP
