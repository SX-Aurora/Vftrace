#include <Python.h>
#include "structmember.h"

#include <stdio.h>

#include "thread_types.h"
#include "threadstack_types.h"
#include "stack_types.h"
#include "profiling_types.h"
#include "profiling.h"
#include "callprofiling.h"
#include "vftrace_state.h"
#include "threadstacks.h"
#include "vftr_initialize.h"

#define PY_MONITORING_EVENT_PY_START 0
#define PY_MONITORING_EVENT_PY_RETURN 2

//#define _PyCFunction_CAST(func) \
//    _Py_CAST(PyCFunction, _Py_CAST(void(*)(void), (func)))

static PyObject *init_vftrace (PyObject *self);

char *repr (PyObject *o) {
   PyObject *r = PyObject_Repr(o);
   PyObject *s = PyUnicode_AsEncodedString(r, "utf-8", "~E~");
   char *out = PyBytes_AS_STRING(s);
   Py_XDECREF(r);
   Py_XDECREF(s);
   return out;
}

PyObject *pystart_callback (PyObject *self, PyObject *const *args, Py_ssize_t size) {
   //printf ("Entry callback\n");
   //printf ("Is code object? %d\n", PyCode_Check(args[0]));
   long long function_entry_time_begin = vftr_get_runtime_nsec();
   PyCodeObject *code = (PyCodeObject *)args[0];
   //printf ("Argcount: %d\n", code->co_argcount);
   //printf ("Name: %s\n", repr(code->co_name));
   //printf ("Filename: %s\n", repr(code->co_filename));
   //printf ("Qualname: %s\n", repr(code->co_qualname));
   //printf ("Line number: %d\n", code->co_firstlineno);

   const char *func_name = repr(code->co_name);
   uint64_t pseudo_addr = (uint64_t)code;
   
   thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
   threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);

   vftr_stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
   profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);

   my_threadstack = vftr_update_threadstack_region (my_threadstack, my_thread,
                                                    (uintptr_t)pseudo_addr, func_name,
                                                    &vftrace, false);
   my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
   my_profile = vftr_get_my_profile(my_stack, my_thread);
   vftr_accumulate_callprofiling(&(my_profile->callprof), 1, 0);
   Py_RETURN_NONE;
}

PyObject *pyreturn_callback (PyObject *self, PyObject *const *args, Py_ssize_t size) {
  //printf ("Return callback\n");
  //printf ("Is code object? %d\n", PyCode_Check(args[0])); 
  long long function_exit_time_begin = vftr_get_runtime_nsec();
  PyCodeObject *code = (PyCodeObject *)args[0];
  char *func_name = repr(code->co_name);
  uint64_t pseudo_addr = (uint64_t)code;
  if (!strcmp (func_name, "'<module>'")) Py_RETURN_NONE;
  //printf ("Return callback name: %s\n", repr(code->co_name));
  //printf ("Return callback qualname: %s\n", repr(code->co_qualname));
  //printf ("Return callback filename: %s\n", repr(code->co_filename));
  //printf ("Line number: %d\n", code->co_firstlineno);
  thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
  //printf ("Get thread: %d\n", my_thread != NULL); 
  threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
  //printf ("Get threadstack: %d %d\n", my_threadstack != NULL, my_thread->stacklist.stacks != NULL);
  vftr_stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
  profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
  //printf ("Pop from %p\n", my_thread->stacklist);
  vftr_accumulate_callprofiling(&(my_profile->callprof), 0, 0);
  (void)vftr_threadstack_pop(&(my_thread->stacklist));
  //printf ("Pop success\n");
  Py_RETURN_NONE;
}

static PyMethodDef vftraceMethods[] = {
   {
      "startVftrace", (PyCFunction)init_vftrace, METH_NOARGS,
      "start Vftrace",
   },
   {
      "_pystart_callback", _PyCFunction_CAST(pystart_callback),
      METH_FASTCALL, NULL,
   },
   {
      "_pyreturn_callback", _PyCFunction_CAST(pyreturn_callback),
      METH_FASTCALL, NULL,
   },
   {NULL, NULL, 0, NULL}
};

static const struct {
   int event;
   const char *callback_method;
} callback_table[] = {
  {PY_MONITORING_EVENT_PY_START, "_pystart_callback"},
  {PY_MONITORING_EVENT_PY_RETURN, "_pyreturn_callback"},
  {0, NULL}
};

static PyObject *init_vftrace (PyObject *self) {
  printf ("Init pyVftrace!\n");
  PyObject *sys = PyImport_ImportModule("sys");
  if (!sys) return NULL;
  printf ("Loaded sys\n");
  PyObject *monitoring = PyObject_GetAttr(sys, PyUnicode_FromString("monitoring"));
  if (!monitoring) return NULL;
  printf ("Loaded monitoring from sys\n");
  int tool_id = 2;
  if (PyObject_CallMethod(monitoring, "use_tool_id", "is", tool_id, "cProfile") == NULL) {
     printf ("Another profiling tool is already active");
     return NULL;
  }
  int all_events = 0;
  for (int i = 0; callback_table[i].callback_method; i++) {
     PyObject *callback = PyObject_GetAttrString(self, callback_table[i].callback_method);
     if (!callback) {
        printf ("Failed to create callback for %s\n", callback_table[i].callback_method);
        return NULL;
     }
     Py_XDECREF(PyObject_CallMethod(monitoring, "register_callback", "iiO", tool_id,
                                    (1 << callback_table[i].event),
                                    callback));
     Py_DECREF(callback);
     all_events |= (1 << callback_table[i].event);
  }
  printf ("Callbacks are set\n");
  if (!PyObject_CallMethod(monitoring, "set_events", "ii", tool_id, all_events)) {
    printf ("Failed to set events\n");
    return NULL;
  }
  printf ("Set events\n");
  Py_DECREF(monitoring);
  vftr_initialize (NULL, NULL);
  Py_RETURN_NONE;
}

static int vftrace_clear_m (PyObject *self) {
   printf ("clear module\n");
}

static void vftrace_free_m (void *self) {
   printf ("free module\n");
}

static struct PyModuleDef vftrace_definition = {
   PyModuleDef_HEAD_INIT,
   .m_name = "vftrace",
   .m_doc = "Python interface for Vftrace",
   //.m_size = ???
   .m_methods = vftraceMethods,
   .m_clear = vftrace_clear_m,
   .m_free = vftrace_free_m,
   //.m_slots = ???
   //.m_traverse = ???
};

//static PyTypeObject cudatraceType = {
//   PyVarObject_HEAD_INIT(NULL, 0);
//   .tp_name = "cudatrace.
//}

PyMODINIT_FUNC PyInit_vftrace(void) {
   Py_Initialize();
   PyObject *thisPy = PyModule_Create(&vftrace_definition);
   //PyModule_AddType(thisPy, &
   return thisPy;
}
