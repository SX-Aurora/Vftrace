#include <Python.h>
#include "frameobject.h"
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
#include "timer.h"
#include "hashing.h"


static PyObject *init_vftrace (PyObject *self);

static PyMethodDef vftraceMethods[] = {
   {
      "startVftrace", (PyCFunction)init_vftrace, METH_NOARGS,
      "start Vftrace",
   },
   {NULL, NULL, 0, NULL}
};

static const struct {
   int event;
   const char *callback_method;
} callback_table[] = {
  //{PY_MONITORING_EVENT_PY_START, "_pystart_callback"},
  //{PY_MONITORING_EVENT_PY_RETURN, "_pyreturn_callback"},
  {0, NULL}
};

char *repr (PyObject *o) {
   PyObject *r = PyObject_Repr(o);
   PyObject *s = PyUnicode_AsEncodedString(r, "utf-8", "~E~");
   char *out = PyBytes_AS_STRING(s);
   Py_XDECREF(r);
   Py_XDECREF(s);
   return out;
}

static int profiler_callback (PyObject *self, PyFrameObject *frame, int what, PyObject *args) {
   long long function_time_begin = vftr_get_runtime_nsec();
   const char *func_name;
   uint64_t pseudo_addr;
   if (what ==  PyTrace_CALL || what == PyTrace_RETURN) {
      //printf ("CALL\n");
      PyCodeObject *fn = (PyCFunctionObject*)frame->f_code;
      func_name = repr(fn->co_name);
      //pseudo_addr = (uint64_t)frame;
      pseudo_addr = vftr_jenkins_murmur_64_hash (strlen(func_name), (uint8_t*)func_name);
      //printf ("%s\n", func_name);
      //printf ("%s: %s @ 0x%lx\n", what == PyTrace_CALL ? "Entry" : "Exit", func_name, pseudo_addr);
   } else if (what ==  PyTrace_C_CALL || what == PyTrace_C_RETURN) {
      //if (PyCFunction_Check(args)) printf ("Is C function!\n");        
      PyCFunctionObject *fn = (PyCFunctionObject*)args;
      PyObject *name = PyUnicode_FromString(fn->m_ml->ml_name);
      func_name = repr(name);
      //pseudo_addr = (uint64_t)args;
      pseudo_addr = vftr_jenkins_murmur_64_hash (strlen(func_name), (uint8_t*)func_name);
      //printf ("C call, func_name: %s\n", func_name);
      //printf ("%s: %s @ 0x%lx\n", what == PyTrace_C_CALL ? "C Entry": "C Exit", func_name, pseudo_addr);
   } else if (what == PyTrace_C_EXCEPTION) {
      printf ("exception\n");
   } else if (what == PyTrace_C_EXCEPTION) {
      printf ("C exception\n");
   } else {
      printf ("UNKNOWN!\n");
   }
   //printf ("func_name: %s\n", func_name);
   //if (!func_name) return 0;
   
   if (what == PyTrace_CALL || what == PyTrace_C_CALL) {
      thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
      threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);

      vftr_stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
      profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);

      my_threadstack = vftr_update_threadstack_region (my_threadstack, my_thread,
                                                       (uintptr_t)pseudo_addr, func_name,
                                                       &vftrace, false);
      my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
      my_profile = vftr_get_my_profile(my_stack, my_thread);
      vftr_accumulate_callprofiling(&(my_profile->callprof), 1, -function_time_begin);
   } else if (what == PyTrace_RETURN || what == PyTrace_C_RETURN) {
      if (!strcmp (func_name, "'<module>'")) return 0;
      thread_t *my_thread = vftr_get_my_thread(&(vftrace.process.threadtree));
      threadstack_t *my_threadstack = vftr_get_my_threadstack(my_thread);
      vftr_stack_t *my_stack = vftrace.process.stacktree.stacks + my_threadstack->stackID;
      profile_t *my_profile = vftr_get_my_profile(my_stack, my_thread);
      vftr_accumulate_callprofiling(&(my_profile->callprof), 0, function_time_begin);
      (void)vftr_threadstack_pop(&(my_thread->stacklist));
   }
   return 0;
}

static PyObject *init_vftrace (PyObject *self) {
  PyEval_SetProfile(profiler_callback, self);
  //printf ("Init pyVftrace!\n");
  //PyObject *sys = PyImport_ImportModule("sys");
  //if (!sys) return NULL;
  //printf ("Loaded sys\n");
  //PyObject *monitoring = PyObject_GetAttr(sys, PyUnicode_FromString("monitoring"));
  //if (!monitoring) return NULL;
  //printf ("Loaded monitoring from sys\n");
  //int tool_id = 2;
  //if (PyObject_CallMethod(monitoring, "use_tool_id", "is", tool_id, "cProfile") == NULL) {
  //   printf ("Another profiling tool is already active");
  //   return NULL;
  //}
  //int all_events = 0;
  //for (int i = 0; callback_table[i].callback_method; i++) {
  //   PyObject *callback = PyObject_GetAttrString(self, callback_table[i].callback_method);
  //   if (!callback) {
  //      printf ("Failed to create callback for %s\n", callback_table[i].callback_method);
  //      return NULL;
  //   }
  //   Py_XDECREF(PyObject_CallMethod(monitoring, "register_callback", "iiO", tool_id,
  //                                  (1 << callback_table[i].event),
  //                                  callback));
  //   Py_DECREF(callback);
  //   all_events |= (1 << callback_table[i].event);
  //}
  //printf ("Callbacks are set\n");
  //if (!PyObject_CallMethod(monitoring, "set_events", "ii", tool_id, all_events)) {
  //  printf ("Failed to set events\n");
  //  return NULL;
  //}
  //printf ("Set events\n");
  //Py_DECREF(monitoring);
  vftr_initialize (NULL, NULL);
  Py_RETURN_NONE;
}

static struct PyModuleDef vftrace_definition = {
   PyModuleDef_HEAD_INIT,
   .m_name = "vftrace",
   .m_doc = "Python interface for Vftrace",
   //.m_size = ???
   .m_methods = vftraceMethods,
   //.m_clear = vftrace_clear_m,
   //.m_free = vftrace_free_m,
   //.m_slots = ???
   //.m_traverse = ???
};

PyMODINIT_FUNC PyInit_vftrace(void) {
   Py_Initialize();
   PyObject *thisPy = PyModule_Create(&vftrace_definition);
   //PyModule_AddType(thisPy, &
   return thisPy;
}

