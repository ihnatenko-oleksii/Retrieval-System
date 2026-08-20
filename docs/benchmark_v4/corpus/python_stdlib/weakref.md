# Python standard library: `weakref`

Official source: https://docs.python.org/3/library/weakref.html

### Weak Reference Objects

Weak reference objects have no methods and no attributes besides ref.__callback__. A weak reference object allows the referent to be obtained, if it still exists, by calling it:

>>> import weakref >>> class Object: ... pass ... >>> o = Object() >>> r = weakref.ref(o) >>> o2 = r() >>> o is o2 True

If the referent no longer exists, calling the reference object returns None:

>>> del o, o2 >>> print(r()) None

Testing that a weak reference object is still live should be done using the expression ref() is not None. Normally, application code that needs to use a reference object should follow this pattern:

# r is a weak reference object o = r() if o is None: # referent has been garbage collected print("Object has been deallocated; can't frobnicate.") else: print("Object is still live!") o.do_something_useful()

Using a separate test for “liveness” creates race conditions in threaded applications; another thread can cause a weak reference to become invalidated before the weak reference is called; the idiom shown above is safe in threaded applications as well as single-threaded applications.

Specialized versions of ref objects can be created through subclassing. This is used in the implementation of the WeakValueDictionary to reduce the memory overhead for each entry in the mapping. This may be most useful to associate additional information with a reference, but could also be used to insert additional processing on calls to retrieve the referent.

This example shows how a subclass of ref can be used to store additional information about an object and affect the value that’s returned when the referent is accessed:

import weakref class ExtendedRef(weakref.ref): def __init__(self, ob, callback=None, /, **annotations): super().__init__(ob, callback) self.__counter = 0 for k, v in annotations.items(): setattr(self, k, v) def __call__(self): """Return a pair containing the referent and the number of times the reference has been called. """ ob = super().__call__() if ob is not None: self.__counter += 1 ob = (ob, self.__counter) return ob

### Example

This simple example shows how an application can use object IDs to retrieve objects that it has seen before. The IDs of the objects can then be used in other data structures without forcing the objects to remain alive, but the objects can still be retrieved by ID if they do.

import weakref _id2obj_dict = weakref.WeakValueDictionary() def remember(obj): oid = id(obj) _id2obj_dict[oid] = obj return oid def id2obj(oid): return _id2obj_dict[oid]

### Finalizer Objects

The main benefit of using finalize is that it makes it simple to register a callback without needing to preserve the returned finalizer object. For instance

>>> import weakref >>> class Object: ... pass ... >>> kenny = Object() >>> weakref.finalize(kenny, print, "You killed Kenny!") <finalize object at ...; for 'Object' at ...> >>> del kenny You killed Kenny!

The finalizer can be called directly as well. However the finalizer will invoke the callback at most once.

>>> def callback(x, y, z): ... print("CALLBACK") ... return x + y + z ... >>> obj = Object() >>> f = weakref.finalize(obj, callback, 1, 2, z=3) >>> assert f.alive >>> assert f() == 6 CALLBACK >>> assert not f.alive >>> f() # callback not called because finalizer dead >>> del obj # callback not called because finalizer dead

You can unregister a finalizer using its detach() method. This kills the finalizer and returns the arguments passed to the constructor when it was created.

>>> obj = Object() >>> f = weakref.finalize(obj, callback, 1, 2, z=3) >>> f.detach() (<...Object object ...>, <function callback ...>, (1, 2), {'z': 3}) >>> newobj, func, args, kwargs = _ >>> assert not f.alive >>> assert newobj is obj >>> assert func(*args, **kwargs) == 6 CALLBACK

Unless you set the atexit attribute to False, a finalizer will be called when the program exits if it is still alive. For instance

>>> obj = Object() >>> weakref.finalize(obj, print, "obj dead or exiting") <finalize object at ...; for 'Object' at ...> >>> exit() obj dead or exiting

### Comparing finalizers with __del__() methods

Suppose we want to create a class whose instances represent temporary directories. The directories should be deleted with their contents when the first of the following events occurs:

the object is garbage collected,

the object’s remove() method is called, or

the program exits.

We might try to implement the class using a __del__() method as follows:

class TempDir: def __init__(self): self.name = tempfile.mkdtemp() def remove(self): if self.name is not None: shutil.rmtree(self.name) self.name = None @property def removed(self): return self.name is None def __del__(self): self.remove()

Starting with Python 3.4, __del__() methods no longer prevent reference cycles from being garbage collected, and module globals are no longer forced to None during interpreter shutdown. So this code should work without any issues on CPython.

However, handling of __del__() methods is notoriously implementation specific, since it depends on internal details of the interpreter’s garbage collector implementation.

A more robust alternative can be to define a finalizer which only references the specific functions and objects that it needs, rather than having access to the full state of the object:

class TempDir: def __init__(self): self.name = tempfile.mkdtemp() self._finalizer = weakref.finalize(self, shutil.rmtree, self.name) def remove(self): self._finalizer() @property def removed(self): return not self._finalizer.alive

Defined like this, our finalizer only receives a reference to the details it needs to clean up the directory appropriately. If the object never gets garbage collected the finalizer will still be called at exit.

The other advantage of weakref based finalizers is that they can be used to register finalizers for classes where the definition is controlled by a third party, such as running code when a module is unloaded:

import weakref, sys def unloading_module(): # implicit reference to the module globals from the function body weakref.finalize(sys.modules[__name__], unloading_module)

Note

If you create a finalizer object in a daemonic thread just as the program exits then there is the possibility that the finalizer does not get called at exit. However, in a daemonic thread atexit.register(), try: ... finally: ... and with: ... do not guarantee that cleanup occurs either.
