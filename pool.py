pool = ThreadPool(4)


In [17]: pool.map(match,[it for it in enumerate(linspace(0,1,10))])
Out[17]: [None, None, None, None, None, None, None, None, None, None]

In [18]: def match(it):
    ...:     (i,a) = it
    ...:     print 'i',i,'a',a
    ...:     agents[i].squarea(a)

In [19]: pool.map(match,[it for it in enumerate(linspace(0,1,10))])
i 0 a 0.0
i 1i  a2  0.111111111111a 0.222222222222

ii 3 4 a a  0.444444444444
0.333333333333
ii 5  a 60.555555555556
 a 0.666666666667 
i i7 a  0.777777777778
8 a 0.888888888889
i 9 a 1.0
Out[19]: [None, None, None, None, None, None, None, None, None, None]

In [20]: [it for it in enumerate(linspace(0,1,10))]
Out[20]: 
[(0, 0.0),
 (1, 0.1111111111111111),
 (2, 0.22222222222222221),
 (3, 0.33333333333333331),
 (4, 0.44444444444444442),
 (5, 0.55555555555555558),
 (6, 0.66666666666666663),
 (7, 0.77777777777777768),
 (8, 0.88888888888888884),
 (9, 1.0)]

In [21]: aa=[it for it in enumerate(linspace(0,1,10))]

In [22]: aa[1]
Out[22]: (1, 0.1111111111111111)

In [23]: i,a=aa[1]

In [24]: i
Out[24]: 1

In [25]: a
Out[25]: 0.1111111111111111

In [26]: def match(it):
    ...:     (i,a) = it
    ...:     print 'i',i,'a',a,'\n'
    ...:     agents[i].squarea(a)

In [27]: def match(it):
    ...:     i,a = it
    ...:     print 'i',i,'a',a,'\n'
    ...:     agents[i].squarea(a)

In [28]: pool.map(match,[it for it in enumerate(linspace(0,1,10))])
i 0 a 0.0 

i 1i a 2 0.111111111111 i a

 3 0.222222222222 i a

  0.333333333333 4 

a 0.444444444444 

i 5i  a6  0.555555555556i a  
7 0.666666666667 

a 
0.777777777778 
i
 8 a i 0.8888888888899  
a 1.0
 

Out[28]: [None, None, None, None, None, None, None, None, None, None]

In [29]: [it for it in enumerate(linspace(0,1,10))]
Out[29]: 
[(0, 0.0),
 (1, 0.1111111111111111),
 (2, 0.22222222222222221),
 (3, 0.33333333333333331),
 (4, 0.44444444444444442),
 (5, 0.55555555555555558),
 (6, 0.66666666666666663),
 (7, 0.77777777777777768),
 (8, 0.88888888888888884),
 (9, 1.0)]

In [30]: match(4,3)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-30-5f77d84eb15c> in <module>()
----> 1 match(4,3)

TypeError: match() takes exactly 1 argument (2 given)

In [31]: match((4,3))
i 4 a 3 


In [32]: match((4,3))
i 4 a 3 


In [33]: match((6,3))
i 6 a 3 


In [34]: match((6,3.3333333))
i 6 a 3.3333333 


In [35]: pool.map(match,[(i,a) for (i,a) in enumerate(linspace(0,1,10))])
i 0i  a 1 0.0 a

 0.111111111111 
i
 2 ia 0.222222222222  
i3 
 4 a 0.333333333333a  

0.444444444444 

i 5 a 0.555555555556 i

 6 ai 0.666666666667  

7 a 0.777777777778 
i
 8 ia  0.888888888889 
9
 a 1.0 

Out[35]: [None, None, None, None, None, None, None, None, None, None]

In [36]: agents[0].alpha
Out[36]: array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

In [37]: agents[1].alpha
Out[37]: 
array([ 0.00015242,  0.00015242,  0.00015242,  0.00015242,  0.00015242,
        0.00015242,  0.00015242,  0.00015242,  0.00015242,  0.00015242])

In [38]: agents[3].alpha
Out[38]: 
array([ 0.22222222,  0.22222222,  0.22222222,  0.22222222,  0.22222222,
        0.22222222,  0.22222222,  0.22222222,  0.22222222,  0.22222222])

In [39]: agents = [agent(alpha=y) for y in range(10)]

In [40]: agents[3].alpha
Out[40]: array([ 3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.,  3.])

In [41]: pool.map(match,[(i,a) for (i,a) in enumerate(linspace(1,11,10))])
i 0i  a 1 1.0 a

 2.11111111111 
i
 2 ia 3.22222222222  
i3 
 4 a 4.33333333333a  

5.44444444444i  
5 
a 6.55555555556 
i
 6 a 7.66666666667i 

 7 a 8.77777777778 

i 8 ia  99.88888888889  

a 11.0 

Out[41]: [None, None, None, None, None, None, None, None, None, None]

In [42]: agents[3].alpha
Out[42]: array([ 13.,  13.,  13.,  13.,  13.,  13.,  13.,  13.,  13.,  13.])

In [43]: agents[0].alpha
Out[43]: array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

In [44]: agents[1].alpha
Out[44]: 
array([ 2.11111111,  2.11111111,  2.11111111,  2.11111111,  2.11111111,
        2.11111111,  2.11111111,  2.11111111,  2.11111111,  2.11111111])

In [45]: linspace(0,1,10)
Out[45]: 
array([ 0.        ,  0.11111111,  0.22222222,  0.33333333,  0.44444444,
        0.55555556,  0.66666667,  0.77777778,  0.88888889,  1.        ])

In [46]: 0.3333333333*3
Out[46]: 0.9999999999

In [47]: agents = [agent(alpha=y) for y in range(10)]

In [48]: linspace(1,11,10)
Out[48]: 
array([  1.        ,   2.11111111,   3.22222222,   4.33333333,
         5.44444444,   6.55555556,   7.66666667,   8.77777778,
         9.88888889,  11.        ])

In [49]: 4.333333333333*3.0
Out[49]: 12.999999999999

In [50]: def match(it):
    ...:     i,a = it
    ...:     agents[i].squarea(a)

In [51]: agents = [agent(alpha=y) for y in range(10)]

In [52]: pool.map(match,[(i,a) for (i,a) in enumerate(linspace(1,10,10))])
Out[52]: [None, None, None, None, None, None, None, None, None, None]

In [53]: agents[1].alpha
Out[53]: array([ 2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.])

In [54]: agents[2].alpha
Out[54]: array([ 6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.,  6.])

In [55]: def _pickle_method(method):
    ...:     func_name = method.im_func.__name__
    ...:     obj = method.im_self
    ...:     cls = method.im_class
    ...:     return _unpickle_method, (func_name, obj, cls)

In [56]: def _unpickle_method(func_name, obj, cls):
    ...:     for cls in cls.mro():
    ...:         try:
    ...:             func = cls.__dict__[func_name]
    ...:             except KeyError:
    ...:                 pass
    ...:     else:
    ...:         break
    ...:     return func.__get__(obj, cls)
  File "<ipython-input-56-3a441f42f632>", line 5
    except KeyError:
         ^
SyntaxError: invalid syntax


In [57]: def _unpickle_method(func_name, obj, cls):
    ...:     for cls in cls.mro():
    ...:         try:
    ...:             func = cls.__dict__[func_name]
    ...:         except KeyError:
    ...:             pass
    ...:     else:
    ...:         break
    ...:     return func.__get__(obj, cls)
  File "<ipython-input-57-ed34cb7f269a>", line 8
    break
SyntaxError: 'break' outside loop


In [58]: def _unpickle_method(func_name, obj, cls):
    ...:     for cls in cls.mro():
    ...:         try:
    ...:             func = cls.__dict__[func_name]
    ...:         except KeyError:
    ...:             pass
    ...:         else:
    ...:             break
    ...:     return func.__get__(obj, cls)

In [59]: import copy_reg

In [60]: import types

In [61]: copy_reg.pickle(types.MethodType,_pickle_method,_unpickle_method)

In [62]: run 93

In [63]: g=cohort()

In [64]: with open('ggg.pickle','wb') as f:
    ...:     pickle.dump(g,f)

In [65]: e,g=value(state(TG=1),cohort(),N=2)
r=4.48, w=1.02, Tr=-0.03 b=0.12,
K=34.52, L=11.17, K/L=3.09
r=5.62, w=0.98, Tr=-0.03 b=0.12,
K=34.84, L=11.47, K/L=3.04
Economy Not Converged in 2 iterations with 0.01
Duration: 0:00:44.184000
<string>:252: RuntimeWarning: invalid value encountered in double_scalars

In [66]: with open('ggg.pickle','wb') as f:
    ...:     pickle.dump(g,f)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-66-fa5bb4495154> in <module>()
      1 with open('ggg.pickle','wb') as f:
----> 2     pickle.dump(g,f)

C:\Anaconda\lib\pickle.pyc in dump(obj, file, protocol)
   1368 
   1369 def dump(obj, file, protocol=None):
-> 1370     Pickler(file, protocol).dump(obj)
   1371 
   1372 def dumps(obj, protocol=None):

C:\Anaconda\lib\pickle.pyc in dump(self, obj)
    222         if self.proto >= 2:
    223             self.write(PROTO + chr(self.proto))
--> 224         self.save(obj)
    225         self.write(STOP)
    226 

C:\Anaconda\lib\pickle.pyc in save(self, obj)
    284         f = self.dispatch.get(t)
    285         if f:
--> 286             f(self, obj) # Call unbound method with explicit self
    287             return
    288 

C:\Anaconda\lib\pickle.pyc in save_inst(self, obj)
    723             stuff = getstate()
    724             _keep_alive(stuff, memo)
--> 725         save(stuff)
    726         write(BUILD)
    727 

C:\Anaconda\lib\pickle.pyc in save(self, obj)
    284         f = self.dispatch.get(t)
    285         if f:
--> 286             f(self, obj) # Call unbound method with explicit self
    287             return
    288 

C:\Anaconda\lib\pickle.pyc in save_dict(self, obj)
    647 
    648         self.memoize(obj)
--> 649         self._batch_setitems(obj.iteritems())
    650 
    651     dispatch[DictionaryType] = save_dict

C:\Anaconda\lib\pickle.pyc in _batch_setitems(self, items)
    661             for k, v in items:
    662                 save(k)
--> 663                 save(v)
    664                 write(SETITEM)
    665             return

C:\Anaconda\lib\pickle.pyc in save(self, obj)
    284         f = self.dispatch.get(t)
    285         if f:
--> 286             f(self, obj) # Call unbound method with explicit self
    287             return
    288 

C:\Anaconda\lib\pickle.pyc in save_list(self, obj)
    598 
    599         self.memoize(obj)
--> 600         self._batch_appends(iter(obj))
    601 
    602     dispatch[ListType] = save_list

C:\Anaconda\lib\pickle.pyc in _batch_appends(self, items)
    613         if not self.bin:
    614             for x in items:
--> 615                 save(x)
    616                 write(APPEND)
    617             return

C:\Anaconda\lib\pickle.pyc in save(self, obj)
    304             reduce = getattr(obj, "__reduce_ex__", None)
    305             if reduce:
--> 306                 rv = reduce(self.proto)
    307             else:
    308                 reduce = getattr(obj, "__reduce__", None)

C:\Anaconda\lib\copy_reg.pyc in _reduce_ex(self, proto)
     75     except AttributeError:
     76         if getattr(self, "__slots__", None):
---> 77             raise TypeError("a class that defines __slots__ without "
     78                             "defining __getstate__ cannot be pickled")
     79         try:

TypeError: a class that defines __slots__ without defining __getstate__ cannot be pickled

In [67]: import math

In [68]: pool.map(math.pow,[1,2,3],[4,5,6])
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-68-e1131ddde199> in <module>()
----> 1 pool.map(math.pow,[1,2,3],[4,5,6])

C:\Anaconda\lib\multiprocessing\pool.pyc in map(self, func, iterable, chunksize)
    249         '''
    250         assert self._state == RUN
--> 251         return self.map_async(func, iterable, chunksize).get()
    252 
    253     def imap(self, func, iterable, chunksize=1):

C:\Anaconda\lib\multiprocessing\pool.pyc in map_async(self, func, iterable, chunksize, callback)
    312 
    313         task_batches = Pool._get_tasks(func, iterable, chunksize)
--> 314         result = MapResult(self._cache, chunksize, len(iterable), callback)
    315         self._taskqueue.put((((result._job, i, mapstar, (x,), {})
    316                               for i, x in enumerate(task_batches)), None))

C:\Anaconda\lib\multiprocessing\pool.pyc in __init__(self, cache, chunksize, length, callback)
    588             del cache[self._job]
    589         else:
--> 590             self._number_left = length//chunksize + bool(length % chunksize)
    591 
    592     def _set(self, i, success_result):

TypeError: unsupported operand type(s) for //: 'int' and 'list'

In [69]: import pathos.multiprocessing as mp
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
<ipython-input-69-d6c72daa621d> in <module>()
----> 1 import pathos.multiprocessing as mp

ImportError: No module named pathos.multiprocessing

In [70]: p=mp.ProcessingPool(4)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
<ipython-input-70-4f34a31f26de> in <module>()
----> 1 p=mp.ProcessingPool(4)

NameError: name 'mp' is not defined

