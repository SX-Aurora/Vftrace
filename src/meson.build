install_headers('vftrace.h')

libvftr_mpi = static_library('vftr_mpi', [
                              'vftr_sync_messages.c',
                              'vftr_async_messages.c',
                              'vftr_mpi_utils.c',
                              'vftr_mpi_environment.c',
                              'vftr_mpi_environment_F.c',
                              'vftr_mpi_environment_F08.c',
                              'vftr_mpi_point2point.c',
                              'vftr_mpi_point2point_F.c',
                              'vftr_mpi_collective.c',
                              'vftr_mpi_collective_F.c',
                              'vftr_mpi_onesided.c',
                              'vftr_mpi_onesided_F.c',
                              'vftr_mpi_testwait.c',
                              'vftr_mpi_testwait_F.c',
                              'vftr_mpi_buf_addr_const.c',
                              'vftr_mpi_buf_addr_const_F.F90',
                            ])

libvftr_mpiwrap = static_library('vftr_mpiwrap', [
                                  'vftr_mpi_environment_cwrap.c',
                                  'vftr_mpi_environment_c2F.F90',
                                  'vftr_mpi_environment_Fwrap.F90',
                                  'vftr_mpi_environment_c2F08.F90',
                                  'vftr_mpi_environment_F08wrap.F90',
                                  'vftr_mpi_point2point_cwrap.c',
                                  'vftr_mpi_point2point_c2F.F90',
                                  'vftr_mpi_point2point_Fwrap.F90',
                                  'vftr_mpi_collective_cwrap.c',
                                  'vftr_mpi_collective_c2F.F90',
                                  'vftr_mpi_collective_Fwrap.F90',
                                  'vftr_mpi_onesided_cwrap.c',
                                  'vftr_mpi_onesided_c2F.F90',
                                  'vftr_mpi_onesided_Fwrap.F90',
                                  'vftr_mpi_testwait_cwrap.c',
                                  'vftr_mpi_testwait_c2F.F90',
                                  'vftr_mpi_testwait_Fwrap.F90'
                                ],
                                 c_args : '-finstrument-functions',
                                 fortran_args : '-finstrument-functions')

libvftr_pause = static_library('vftr_pause', [
                                'vftr_pause.c'
                              ],
                               c_args : '-finstrument-functions')

# Use link_whole to link the complete content of the static libraries into shared library else the linker will not add them as the symbols in the static library are not (directly) used.
# Add explicit (Fortran) MPI dependency to correctly link against MPI library without changing the link language, thus, the MPI wrapper.
libvftrace = library('vftrace', [
                      'demangle.cpp',
                      'vftr_scenarios.c',
                      'vftr_symbols.c',
                      'vftr_hwcounters.c',
                      'vftr_hooks.c',
                      'vftr_setup.c',
                      'vftr_filewrite.c',
                      'vftr_fileutils.c',
                      'vftr_hashing.c',
                      'vftr_sorting.c',
                      'vftr_stacks.c',
                      'vftr_user_stack.c',
                      'vftr_functions.c',
                      'vftr_regions.c',
                      'vftr_timer.c',
                      'tinyexpr.c',
                      'vftr_regex.c',
                      'vftr_environment.c',
                      'vftr_signals.c',
                      'vftr_loadbalance.c',
                      'vftrace_mod.F90'
                    ],
                     link_whole : [
                       libvftr_pause,
                       libvftr_mpi,
                       libvftr_mpiwrap
                     ],
                     dependencies : mpi_dep,
                     install : true)

libvftr_dlopen = library('vftr_dlopen', ['vftr_dlopen.c'],
                         dependencies : dldep,
                         install : true)

inc_dep = include_directories('.')
