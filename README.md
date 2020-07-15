# Vftrace

## About

Vftrace (visual ftrace) is a performance profiling library with a focus on applications in high-performance computing (HPC). It is compatible with C and Fortran. It supports NEC's SX Aurora vector architecture and x86 architectures. On top of that, Vftrace can be linked against Veperf (SX Aurora) or PAPI (x86) to measure hardware counters. These hardware counters can be transformed into user-defined performance observables, such as GFLOPS or vector length, using json configuration files.
Vftrace produces an overview of the function calls appearing during an application's runtime and registers the call number and the time spent in the code parts. If hardware observables are defined, their value is also reported.
The generated profile can be visualized and analyzed with the Vfview tool.

## Usage

Vftrace requires that your application has instrumendet function calls. These are enabled with a compiler flag, most commonly known as `-finstrument-functions`, as supported by the GNU, Intel and NEC compilers.
After compiling, you must link your application against `libvftrace`, either statically or dynamically.
The application can then be run in the usual way. In the default setting, a text file is created containing a run-time profile of the application.

## Prerequisites & Installation 

Vftrace is written in C. For the Fortran interface, there is also some Fortran code.
Vftrace is built using the standard autotools toolchain. For more information, consult the `INSTALL` file included in this repository.

We recommend to compile your application with the same compiler you used to compile Vftrace. It has to support function instrumentation. To our knowledge, this is given for the following list of compilers:
  - GNU
  - Intel
  - NEC

For hardware counter support on x86 systems, PAPI is required: https://icl.utk.edu/papi/

## Basic Principle

Vftrace uses the Cygnus function hooks:

```C
   void __cyg_profile_func_enter (void *function_addr, void *caller_addr);
   void __cyg_profile_func_exit  (void *function_addr, void *caller_addr);
```

These functions deal as trampolines in the application binary which the programmer can intercept. They need to be enabled by the compiler using the `-finstrument-functions` option. 
The arguments of these functions are the addresses of the symbols of the file mapped into virtual address space (`/proc/<pid>/maps`). At initialization, Vftrace reads the ELF file of the executable, as well as its dependencies, to assign names to these symbols. However, not the entire table is used during run-time, since this would imply a too large overhead. Instead, Vftrace dynamically allocates function stacks. A stack element is a path from the `init` or `main` function to each individual function which has been registered during run-time, including the functions in between.
For each function entry and exit, the current stack is measured and registered.

## MPI Profiling

Vftrace has wrappers to MPI function calls. These wrappers are instrumented and call a `vftr_MPI` version of the function. The wrappers for C and Fortran both call the same `vftr_MPI` routine to reduce code duplication and enable easier maintenance.
```C
   int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
                int dest, int tag, MPI_Comm comm) {
      return vftr_MPI_Send(buf, count, datatype, dest, tag, comm);
   }
```

In these `vftr_MPI`-functions the communication is executed by calling the corresponding `PMPI` function:
```C
   int vftr_MPI_Send(const void *buf, int count, MPI_Datatype datatype,
                     int dest, int tag, MPI_Comm comm) {
   
      // disable profiling based on the Pcontrol level
      if (vftrace_Pcontrol_level == 0) {
         return PMPI_Send(buf, count, datatype, dest, tag, comm);
      } else {
         long long tstart = vftr_get_runtime_usec();
         int retVal = PMPI_Send(buf, count, datatype, dest, tag, comm);
         long long tend = vftr_get_runtime_usec();
   
         vftr_store_sync_message_info(send, count, datatype, dest, tag, comm, tstart, tend);
   
         return retVal;
      }
   }
```
The `PMPI_` symbols do the same as their `MPI_` counterpart. This way, the MPI functions as used by the application are instrumented. The functionality inside the wrapper enables in-depth MPI sampling. The vftrace MPI-wrapper record which ranks are communicating (who sends, who receives), the message size, message type, and communication time.
Non blocking communication is sampled by registering the non blocking call's request and checking for completion from time to time in the background.

## Hardware Observables

PAPI has a list of native hardware events, which depends on the actual CPU you are working on. You can use the names of the native events to construct your own hardware observables, using a json input script. For example, to measure the Level 1 cache hit rate, you can use the following script:
```
{
  "scenario_name": "DCACHE",
  "CPU Type": "Skylake",
  "counters": [
    {
      "papi_name": "PERF_COUNT_HW_CACHE_L1D:ACCESS",
      "symbol": "f1"
    },
    {
      "papi_name": "PERF_COUNT_HW_CACHE_L1D:MISS",
      "symbol": "f2"
    }
  ],
  "observables": [
    {
      "name": "L1D cache hit ratio",
      "formula": "f1 * 100 / (f1 + f2)",
      "default": "0",
      "format": [{
		"unit": "%",
		"spec": "4.1",
		"group": "D-cache",
		"column1": "L1D",
		"column2": "%Hit"}]
    }
  ]
}
```
Here, we register two PAPI events, for cache hits and misses, and assign a variable to each of them. With these variables, we define the observable "L1D cache hit ratio". In the text output, its value will appear next to the run times for each function stack.

## Graphical User Interface

The graphical visualization tool for Vtrace profiles, Vfview, is located at https://github.com/SX-Aurora/Vfview.

## Authors

Vftrace was originally conveived by Jan Boerhout.
The main authors are:
  - Felix Uhl (felix.uhl@emea.nec.com)
  - Christian Weiss (christian.weiss@emea.nec.com)

## Third Party Tools

Vftrace uses the following open-source third party tools:

  - The json parser "jsmn" by Serge Zaitsev (https://github.com/zserge/jsmn). It is used to read in the hardware scenario files.
  - Lewis van Winkle's "tinyexpr" (https://github.com/codeplea/tinyexpr). It is used to parse the formula strings which define hardware observables.
  - Adapted Jenkins (https://en.wikipedia.org/wiki/Jenkins_hash_function) and
    Murmur3 (https://en.wikipedia.org/wiki/MurmurHash) hash functions originally published under the creative common license (https://creativecommons.org/licenses/by-sa/3.0/).
    The hashes are used to identify individual stacks among different MPI-ranks.

## Licensing

Vftrace is licensed under The GNU general public license (GPL), which means that you are free to copy and modify the source code under the condition that you keep the license.

## How to Contribute

You are free to clone or fork this repository as you like. If you wish to make a contribution, please open up a pull request. Consult the `CODEOWNERS` file for more information about contact persons for specific parts of the code.
If you find a bug in Vftrace or have an idea for an improvement, please submit an issue on github.

## FAQ

### Which languages are supported besides C and Fortran?

In principle, Vftrace can support every compiled language which can create a function hook like `__cyg_profile_func_enter`. As of now, the only languages we know of that support this feature are
C, C++ and Fortran. On top of that, the executable has to be in the ELF format and Vftrace must be able to parse the symbol names out of it. This is still not optimal for C++, which is the reason why it is not yet officially supported.

### Does Vftrace support OpenMP?

Although there are code passages which refer to OpenMP, Vftrace does not yet officially support OpenMP. This is because it is still an open question how the dynamic creation of threads can be combined with the function-stack structure of Vfrace.
