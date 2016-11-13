/*
 * ccallback
 *
 * Callback function interface, supporting
 *
 * (1) pure Python functions
 * (2) plain C functions wrapped in PyCapsules (with Cython CAPI style signatures)
 * (3) ctypes function pointers
 * (4) cffi function pointers
 *
 * This is done avoiding magic or code generation, so you need to write some
 * boilerplate code manually.
 *
 * For an example see `scipy/_lib/src/_test_ccallback.c`.
 */


#ifndef CCALLBACK_H_
#define CCALLBACK_H_


#include <Python.h>
#include <setjmp.h>

/* Default behavior */
#define CCALLBACK_DEFAULTS 0x0
/* Whether calling ccallback_obtain is enabled */
#define CCALLBACK_OBTAIN   0x1
/* Deal with also other input objects than LowLevelCallable.
 * Useful for maintaining legacy behavior.
 */
#define CCALLBACK_PARSE    0x2


typedef struct ccallback ccallback_t;

struct ccallback {
    /* Pointer to a C function to call. NULL if none. */
    void *c_function;
    /* Pointer to a Python function to call (refcount is owned). NULL if none. */
    PyObject *py_function;
    /* Additional data pointer provided by the user. */
    void *user_data;
    /* Index of the function signature selected */
    int signature_index;
    /* setjmp buffer to jump to on error */
    jmp_buf error_buf;
    /* Previous callback, for TLS reentrancy */
    ccallback_t *prev_callback;

    /* Unused variables that can be used by the thunk etc. code for any purpose */
    long info;
    void *info_p;
};


/*
 * Thread-local storage
 */

#if defined(__GNUC__) && (__GNUC__ > 4 || (__GNUC__ == 4 && (__GNUC_MINOR__ >= 4)))

static __thread ccallback_t *_active_ccallback = NULL;

static void *ccallback__get_thread_local(void)
{
    return (void *)_active_ccallback;
}

static int ccallback__set_thread_local(void *value)
{
    _active_ccallback = value;
    return 0;
}

/*
 * Obtain a pointer to the current ccallback_t structure.
 */
static ccallback_t *ccallback_obtain(void)
{
    return (ccallback_t *)ccallback__get_thread_local();
}

#elif defined(_MSC_VER)

static __declspec(thread) ccallback_t *_active_ccallback = NULL;

static void *ccallback__get_thread_local(void)
{
    return (void *)_active_ccallback;
}

static int ccallback__set_thread_local(void *value)
{
    _active_ccallback = value;
    return 0;
}

/*
 * Obtain a pointer to the current ccallback_t structure.
 */
static ccallback_t *ccallback_obtain(void)
{
    return (ccallback_t *)ccallback__get_thread_local();
}

#else

/* Fallback implementation with Python thread API */

static void *ccallback__get_thread_local(void)
{
    PyObject *local_dict, *capsule;
    void *callback_ptr;

    local_dict = PyThreadState_GetDict();
    if (local_dict == NULL) {
        Py_FatalError("scipy/ccallback: failed to get local thread state");
    }

    capsule = PyDict_GetItemString(local_dict, "__scipy_ccallback");
    if (capsule == NULL) {
        return NULL;
    }

    callback_ptr = PyCapsule_GetPointer(capsule, NULL);
    if (callback_ptr == NULL) {
        Py_FatalError("scipy/ccallback: invalid callback state");
    }

    return callback_ptr;
}

static int ccallback__set_thread_local(void *value)
{
    PyObject *local_dict;

    local_dict = PyThreadState_GetDict();
    if (local_dict == NULL) {
        Py_FatalError("scipy/ccallback: failed to get local thread state");
    }

    if (value == NULL) {
        return PyDict_DelItemString(local_dict, "__scipy_ccallback");
    }
    else {
        PyObject *capsule;
        int ret;

        capsule = PyCapsule_New(value, NULL, NULL);
        if (capsule == NULL) {
            return -1;
        }
        ret = PyDict_SetItemString(local_dict, "__scipy_ccallback", capsule);
        Py_DECREF(capsule);
        return ret;
    }
}

/*
 * Obtain a pointer to the current ccallback_t structure.
 */
static ccallback_t *ccallback_obtain(void)
{
    PyGILState_STATE state;
    ccallback_t *callback_ptr;

    state = PyGILState_Ensure();

    callback_ptr = (ccallback_t *)ccallback__get_thread_local();
    if (callback_ptr == NULL) {
        Py_FatalError("scipy/ccallback: failed to get thread local state");
    }

    PyGILState_Release(state);

    return callback_ptr;
}

#endif


/*
 * Set up callback.
 *
 * Parameters
 * ----------
 * callback : ccallback_t
 *     Callback structure to initialize.
 * signatures : char **
 *     Pointer to a NULL-terminated array of C function signature strings.
 *     The list of signatures should always contain a signature defined in
 *     terms of C basic data types only.
 * callback_obj : PyObject
 *     Object provided by the user. Usually, LowLevelCallback object, or a
 *     Python callable.
 * flags : int
 *     Bitmask of CCALLBACK_* flags.
 *
 * Returns
 * -------
 * success : int
 *     0 if success, != 0 on failure (an appropriate Python exception is set).
 *
 */
static int ccallback_prepare(ccallback_t *callback, char **signatures, PyObject *callback_obj, int flags)
{
    static PyTypeObject *lowlevelcallable_type = NULL;
    PyObject *callback_obj2 = NULL;
    PyObject *capsule = NULL;

    if (lowlevelcallable_type == NULL) {
        PyObject *module;

        module = PyImport_ImportModule("scipy._lib._ccallback");
        if (module == NULL) {
            goto error;
        }

        lowlevelcallable_type = (PyTypeObject *)PyObject_GetAttrString(module, "LowLevelCallable");
        Py_DECREF(module);
        if (lowlevelcallable_type == NULL) {
            goto error;
        }
    }

    if ((flags & CCALLBACK_PARSE) && !PyObject_TypeCheck(callback_obj, lowlevelcallable_type)) {
        /* Parse callback */
        callback_obj2 = PyObject_CallMethod((PyObject *)lowlevelcallable_type,
                                            "_parse_callback", "O", callback_obj);
        if (callback_obj2 == NULL) {
            goto error;
        }

        callback_obj = callback_obj2;

        if (PyCapsule_CheckExact(callback_obj)) {
            capsule = callback_obj;
        }
    }

    if (PyCallable_Check(callback_obj)) {
        /* Python callable */
        callback->py_function = callback_obj;
        Py_INCREF(callback->py_function);
        callback->c_function = NULL;
        callback->user_data = NULL;
        callback->signature_index = -1;
    }
    else if (PyObject_TypeCheck(callback_obj, lowlevelcallable_type) &&
             PyCallable_Check(PyTuple_GET_ITEM(callback_obj, 0))) {
        /* Python callable in LowLevelCallable */
        callback->py_function = PyTuple_GET_ITEM(callback_obj, 0);
        Py_INCREF(callback->py_function);
        callback->c_function = NULL;
        callback->user_data = NULL;
        callback->signature_index = -1;
    }
    else if (capsule != NULL ||
             (PyObject_TypeCheck(callback_obj, lowlevelcallable_type) &&
              PyCapsule_CheckExact(PyTuple_GET_ITEM(callback_obj, 0)))) {
        /* PyCapsule in LowLevelCallable (or parse result from above) */
        void *ptr, *user_data;
        char **sig;
        const char *name;

        if (capsule == NULL) {
            capsule = PyTuple_GET_ITEM(callback_obj, 0);
        }

        name = PyCapsule_GetName(capsule);
        if (PyErr_Occurred()) {
            goto error;
        }
        
        callback->signature_index = 0;
        for (sig = signatures; *sig != NULL; ++sig, ++callback->signature_index) {
            if (name && strcmp(name, *sig) == 0) {
                break;
            }
        }

        ptr = PyCapsule_GetPointer(capsule, *sig);
        if (ptr == NULL) {
            PyErr_SetString(PyExc_ValueError, "Invalid function signature in PyCapsule");
            goto error;
        }

        user_data = PyCapsule_GetContext(capsule);
        if (PyErr_Occurred()) {
            goto error;
        }

        callback->py_function = NULL;
        callback->c_function = ptr;
        callback->user_data = user_data;
    }
    else {
        PyErr_SetString(PyExc_ValueError, "invalid callable given");
        goto error;
    }

    if (flags & CCALLBACK_OBTAIN) {
        callback->prev_callback = ccallback__get_thread_local();
        if (ccallback__set_thread_local((void *)callback) != 0) {
            goto error;
        }
    }
    else {
        callback->prev_callback = NULL;
    }

    Py_XDECREF(callback_obj2);
    return 0;

error:
    Py_XDECREF(callback_obj2);
    return -1;
}


/*
 * Tear down callback.
 *
 * Parameters
 * ----------
 * callback : ccallback_t
 *     A callback structure, previously initialized by ccallback_prepare
 *
 */
static int ccallback_release(ccallback_t *callback)
{
    Py_XDECREF(callback->py_function);
    callback->c_function = NULL;
    callback->py_function = NULL;

    if (callback->prev_callback != NULL) {
        if (ccallback__set_thread_local(callback->prev_callback) != 0) {
            return -1;
        }
    }
    callback->prev_callback = NULL;

    return 0;
}

#endif /* CCALLBACK_H_ */
