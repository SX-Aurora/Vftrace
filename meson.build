project('vftrace',
        ['c', 'cpp', 'fortran'],
        license : 'GPL2+',
        version : '0.0',
        meson_version : '>=0.55.0',
        # see 'meson configure' (in build directory) for all available options
        default_options : [
          'default_library=static',
          'buildtype=debug',
          'fortran_std=none'
        ])

# GCC10 introduced a paradigm-change by defaulting to -fno-common, https://wiki.gentoo.org/wiki/Gcc_10_porting_notes/fno_common.
# Rollback to previous behavior: -fcommon.
add_project_arguments('-fcommon', language : ['c', 'cpp', 'fortran'])

# Fortran
fortran = meson.get_compiler('fortran')
if not fortran.has_argument('-finstrument-functions')
  error('Fortran compiler has no support for "-finstrument-functions"')
endif

# C / C++
cc = meson.get_compiler('c')
libmath_dep = cc.find_library('m', required : false) # Only needed for subproject dependency.
if not cc.has_argument('-finstrument-functions')
  error('C compiler has no support for "-finstrument-functions"')
endif
cpp = meson.get_compiler('cpp')
if not cpp.has_argument('-finstrument-functions')
  error('C++ compiler has no support for "-finstrument-functions"')
endif

dldep = cc.find_library('dl')
mpi_dep = dependency('mpi', language : 'fortran', required : get_option('mpi-profiling'))
if mpi_dep.found ()
  add_project_arguments('-D_MPI', language: ['c', 'cpp', 'fortran'])
  if fortran.get_id() == 'intel'
    add_project_arguments('-DPMPI_MODULE=pmpi_f08', language : 'fortran')
  else
    add_project_arguments('-DPMPI_MODULE=mpi', language : 'fortran')
  endif
endif

if get_option('papi').enabled()
  add_project_arguments('-DHAS_PAPI -I@0@'.format(get_option('papi-dir')), language : ['c', 'cpp'])
endif

if get_option('veperf').enabled()
  add_project_arguments('-DHAS_VEPERF -I@0@'.format(get_option('veperf-dir')), language : ['c', 'cpp'])
  add_project_link_arguments('-L@0@/lib -lveperf_ext'.format(get_option('veperf-dir')))
endif

# Add VMAP offset check using config_test.sh
if get_option('vmap-offset').enabled()
  message('Use VMAP offset')
  add_project_arguments('-D__VMAP_OFFSET', language : 'c')
endif

# Workaround: Install vftrace.mod
meson.add_install_script('install_vftrace_mod.sh', get_option('prefix'), get_option('includedir'))

subdir('src')
subdir('share_src/man/man1')

libvftrace_dep = declare_dependency(include_directories : inc_dep,
                                    dependencies : libmath_dep,
                                    link_with : [libvftrace, libvftr_dlopen])
