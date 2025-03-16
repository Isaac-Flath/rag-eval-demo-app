Isaac Flath

AboutBlogRSS

Subscribe

AboutBlogRSS

Subscribe

# Python Programming Tips

A list of handy tips and tricks when programming in python

## Contents

SetupPurposeComprehensionsDestructuringFastcoreParallel ProcessingDocmentsTesting LAttrDictLoggingHigher Order Functions & CallbacksFilter Map Simple LoggingFile ProcessorFormat and clean SQL fileDecoratorsSilly Simple ExampleUseful ExampleUse on Existing FunctionsInheritanceSilly Simple ExampleUseful ExamplesDundersIterators/Data StreamingSilly Simple ExampleUseful ExampleFile Streaming

# Setup

Show CodeHide Code

[code]

    from functools import partial
    from datetime import datetime
    import logging, string, pandas as pd, sqlparse
    from fastcore.all import *
    from fastcore.docments import *
    from IPython.display import Markdown,display, HTML
    import pandas as pd
    
    from pygments import highlight
    from pygments.lexers import PythonLexer
    from pygments.formatters import HtmlFormatter
    
    def print_function_source(fn):
        fn = print_decorator
        formatter = HtmlFormatter()
        display(HTML('<style type="text/css">{}</style>{}'.format(
            formatter.get_style_defs('.highlight'),
            highlight(inspect.getsource(fn), PythonLexer(), formatter))))
[/code]

# Purpose

The purpose of this is to introduce concepts I believe data scientists could benefit from knowing.

I am assuming that the reader knows the basics of programming. I will cover concepts I frequently see that I think are not used enough or appropriately, regardless of how basic or advanced they may be.

# Comprehensions

Comprehensions in python should be used when possible. They are faster than forloops and require less code when they fit

[code]

    x = [2,3,4,5]
[/code]

[code]

    out=[]
    %timeit for i in range(1000000): out.append(i+1)
[/code]

[code]

    87.2 ms ± 647 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    
[/code]

[code]

    %timeit [i+1 for i in range(1000000)]
[/code]

[code]

    56.8 ms ± 584 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    
[/code]

This is basically special syntax for a forloop, and are useful in a subset of forloops. Basically any time you see the pattern where you initialize something, then modify or build it in the forloop you can likely use a comprehension

[code]

    out = []
    for o in range(5): out.append(o**o)
    out
[/code]

[1, 1, 4, 27, 256]

[code]

    [o**o for o in range(5)]
[/code]

[1, 1, 4, 27, 256]

List comprehensions are most common but you can also do tuple comprehension, set comprehension, dict comprehension, or other data types.

[code]

    set(o**o for o in range(5))
[/code]

{1, 4, 27, 256}

[code]

    {str(o):o**o for o in range(5)}
[/code]

{'0': 1, '1': 1, '2': 4, '3': 27, '4': 256}

A few handy patterns are:

  * Reversing a dictionary
  * Combining lists
  * All unique combos from multiple lists (nested comprehension)

[code]

    adict = {"a":1,"b":2}
    {v:k for k,v in adict.items()}
[/code]

{1: 'a', 2: 'b'}

[code]

    x = [1,2,3,4]
    y = [5,6,7,8]
[/code]

[code]

    [a+b for a,b in zip(x,y)]
[/code]

[6, 8, 10, 12]

[code]

    unique_combos = L((a,b) for a in x for b in y)
    unique_combos
[/code]

(#16) [(1, 5),(1, 6),(1, 7),(1, 8),(2, 5),(2, 6),(2, 7),(2, 8),(3, 5),(3, 6)...]

# Destructuring

Destructured assignments mean to can break up iterables when you assign. This is handy to reduce pointless lines of code.

[code]

    a,b = 5,6
    a,b,c = [],[],{}
[/code]

Another use is to break up lists to create lists where we take all the first elements out into it's own list, and the second elements out into their own lists.

I often see this done with multiple list comprehension, doing `[o[0] for o in [x,y,z]]` to get the first element, then repeating for other elements.

However, we can do this easier with the help of zip and destructured assignments

[code]

    nested_list = [[1,2,3],[4,5,6],[7,8,9]]
    nested_list
[/code]

[[1, 2, 3], [4, 5, 6], [7, 8, 9]]

[code]

    first_elements, second_elements, third_elements = list(zip(*nested_list))
[/code]

[code]

    print(f"{first_elements=}")
    print(f"{second_elements=}")
    print(f"{third_elements=}")
[/code]

[code]

    first_elements=(1, 4, 7)
    second_elements=(2, 5, 8)
    third_elements=(3, 6, 9)
    
[/code]

# Fastcore

Fastcore is a great library to know. It's got a lot of useful features and extensions to the python standard library and it's designed to be used in live environments like jupyter notebooks.

## Parallel Processing

See this blog post

### Docments

Nice way of documenting code concisely and being able to access info from code. It's concise, easy to manipulate to display how you want, and easy to read. I much prefer it over the large numpy style docstrings that are big string blocks

[code]

    from fastcore.docments import *
    
    def distance(pointa:tuple,  # tuple representing the coordinates of the first point (x,y)
                 pointb:tuple=(0,0) # tuple representing the coordinates of the first point (x,y)
                )->float: # float representing distance between pointa and pointb
        '''Calculates the distance between pointa and pointb'''
        edges = np.abs(np.subtract(pointa,pointa))
        distance = np.sqrt((edges**2).sum())
        return distance
[/code]

[code]

    docstring(distance)
[/code]

Calculates the distance between pointa and pointb

[code]

    docments(distance)
[/code]

[code]

    { 'pointa': 'tuple representing the coordinates of the first point (x,y)',
      'pointb': 'tuple representing the coordinates of the first point (x,y)',
      'return': 'float representing distance between pointa and pointb'}
    
[/code]

[code]

    docments(distance,full=True)
[/code]

[code]

    { 'pointa': { 'anno': <class 'tuple'>,
                  'default': <class 'inspect._empty'>,
                  'docment': 'tuple representing the coordinates of the first '
                             'point (x,y)'},
      'pointb': { 'anno': <class 'tuple'>,
                  'default': (0, 0),
                  'docment': 'tuple representing the coordinates of the first '
                             'point (x,y)'},
      'return': { 'anno': <class 'float'>,
                  'default': <class 'inspect._empty'>,
                  'docment': 'float representing distance between pointa and '
                             'pointb'}}
    
[/code]

### Testing

Everyone agrees testing is important. But not all testing is equal. The needs for unit testing the google code base are not the same as the needs a data scientist needs for building and deploying models, libraries, and most software.

Fastcore is a great tool for most of my testing needs. Fast and simple enough that I can add tests as I build and as I am exploring and building models. I want testing to enhance my development workflow, not be something I have to painstakingly build at the end.

Sometimes simple assert statements are sufficient, but there's small annoyances. For example, a small change in type can mean a failed test. Sometimes that change in type should cause a failure, sometimes I'm ok if it's a different type if the values are the same

[code]

    from fastcore.test import *
[/code]

[code]

    test_eq([1,2],(1,2))
[/code]

For floating points it has handy functionality for that, which is very common in data science. For example, we may want `.1 + .1 + .1 == .3` to be true, because they are close enough based on floating point precision

[code]

    .1 + .1 + .1 == .3
[/code]

False

[code]

    test_close(.1 + .1 + .1, .3)
[/code]

We can test that something fails, if there are particular situation we want to ensure raise errors.

[code]

    def _fail(): raise Exception("foobar")
    test_fail(_fail)
[/code]

We can test if 2 lists have the same values, just in different orders (convenient for testing some situations with random mini-batches).

[code]

    a = list(range(5))
    b = a.copy()
    b.reverse()
    test_shuffled(a,b)
[/code]

There's more of course, check out the docs

### L

L is a replacement for a list, but with lots of adding functionality. Some of it are functional programming concepts, some is numpy like stuff, and some is just niceities (like cleaner printing).

[code]

    alist = L(1,2,3,4,3)
[/code]

[code]

    alist.sort()
    alist.sorted()
[/code]

(#5) [1,2,3,3,4]

[code]

    alist.unique()
[/code]

(#4) [1,2,3,4]

[code]

    alist.filter(lambda x: x < 3)
[/code]

(#2) [1,2]

[code]

    alist.map(lambda x: x * 2)
[/code]

(#5) [2,4,6,8,6]

### AttrDict

`Attrdict` is another nice thing from fastcore, that makes dictionaries a bit nicer to use.

[code]

    regdict = {'a':2,'b':3}
    adict = AttrDict({'a':2,'b':3})
[/code]

[code]

    adict
[/code]

[code]

    {'a': 2, 'b': 3}
    
[/code]

[code]

    adict.a
[/code]

2

[code]

    def _fail(): return regdict.a
    test_fail(_fail)
[/code]

# Logging

Logging is super important. if you log stuff as you work properly you can always look back at what was done previously. Sometimes it's hard to tell what's going on as you run and re-run different things. Logging is handy not just in production for debugging, but also as a tool when you are developing. There are many tools to help with logging and visualizing results (for example W&B or tensorboard for deep learning) - but the foundations are good to understand and use too!

[code]

    logging.basicConfig(filename="./mylog.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
[/code]

[code]

    def get_current_time(): return datetime.now().strftime('%Y%m%d_%H%M%S')
[/code]

[code]

    logger.info    (f'{get_current_time()}|This is an info message')
[/code]

[code]

    !head -4 mylog.log
[/code]

[code]

    INFO:root:20221106_111500|This is an info message
    INFO:root:20221106_111521|Starting the model training process
    INFO:root:20221106_111521|Training set has 50 records
    INFO:root:20221106_111521|Validtion set has 70 records
    
[/code]

[code]

    def log_stuff(msg,**kwargs): 
        dt = get_current_time()
        logger.info(f"{dt}|{msg}")
        for k,v in kwargs.items(): logger.info(f"{dt}|{k}={v}")
[/code]

[code]

    log_stuff('this is what I want to log',
              trainig_set='50 records',
              validation_set='70_records')
[/code]

# Higher Order Functions & Callbacks

This is a simple example of what these terms mean:

  * A higher order function is a function that takes a function as an argument
  * A callback is a function that is passed in as an argument to a higher order function

[code]

    def callbackFunc1(s): print('Callback Function 1: Length of the text file is : ', s)
    def callbackFunc2(s): print('Callback Function 2: Length of the text file is : ', s)
    
    def HigherOrderFunction(path, callback):
        with open(path, "r") as f: callback(len(f.read()))
    
    HigherOrderFunction("mylog.log", callbackFunc1)
    HigherOrderFunction("mylog.log", callbackFunc2)
[/code]

[code]

    Callback Function 1: Length of the text file is :  1130
    Callback Function 2: Length of the text file is :  1130
    
[/code]

This is handy in a lot of situations.

## Filter

Filter is a common higher order function.

[code]

    L(1,2,3,4,5).filter(lambda x: x>3)
[/code]

(#2) [4,5]

This is very flexible because we can put filtering logic of any complexity in a function and use that to filter a list of any type.

## Map

Map is another very common higher order function.

[code]

    L(1,2,3,4,5).map(lambda x: x**2)
[/code]

(#5) [1,4,9,16,25]

It is again super flexible because we can apply a function of any complexity to have it be applied and modify each element of the list.

[code]

    L(1,2,3,4,5).map(lambda x: string.ascii_lowercase[x])
[/code]

(#5) ['b','c','d','e','f']

## Simple Logging

We could make a function for logging, where we can pass a function in that we want to use for logging (ie info vs warning).

[code]

    def log_stuff(msg,fn=logger.info,**kwargs): 
        dt = get_current_time()
        fn(f"{dt}|{msg}")
        for k,v in kwargs.items(): fn(f"{dt}|{k}={v}")
[/code]

[code]

    log_stuff('abcd',a=1,b=55)
[/code]

[code]

    !tail -3 mylog.log
[/code]

[code]

    INFO:root:20221106_193211|abcd
    INFO:root:20221106_193211|a=1
    INFO:root:20221106_193211|b=55
    
[/code]

[code]

    log_stuff('something might be awry',fn=logger.critical,a=1,b=55)
[/code]

[code]

    !tail -3 mylog.log
[/code]

[code]

    CRITICAL:root:20221106_193211|something might be awry
    CRITICAL:root:20221106_193211|a=1
    CRITICAL:root:20221106_193211|b=55
    
[/code]

## File Processor

You can also make a generic file processor that you can pass callbacks to. This file processor can include log statements to log what you're doing, so you can minimize repeating lots of code. For now, we'll do a simple processor, and callbacks to clean and format a messy sql file.

[code]

    def process_file(fpath,callbacks): 
        with open(fpath, "r") as f: contents = f.read()
        for callback in callbacks: contents = callback(contents)
        return contents
[/code]

## Format and clean SQL file

[code]

    sql_formatter_cb = partial(sqlparse.format,
                    strip_comments=True,comma_first=True,
                    keyword_case='upper', identifier_case='lower',
                    reindent=True, indent_width=4,)
    
    
    
    qrys = process_file('test.sql',[sql_formatter_cb,sqlparse.split])
[/code]

[code]

    def sql_pprint(sql): display(Markdown(f"```sql\n\n{sql}\n\n```"))
    for qry in qrys: sql_pprint(qry)
[/code]

[code]

    SELECT top 25 *
    FROM some_table;
    
    
[/code]

[code]

    SELECT count(1)
    FROM another TABLE ;
    
    
[/code]

[code]

    SELECT date_time
         , mbr_id
         , transactions
         , count(1)
    FROM table3
    WHERE date_time > '2021-02-02'
    GROUP BY 1
           , 2
           , 3;
    
    
[/code]

# Decorators

Decorators give you a way to add the same functionality to many functions (like inheritance does for classes). You typically use decorator using the `@` syntax, which modified the function.

## Silly Simple Example

[code]

    def add_another(func):
        def wrapper(number):
            print(f"The decorator took over!")
            print(f"I could log the original number ({number}) here!")
            print(f"Or I could log the original answer ({func(number)}) here!")
            return func(number) + 1
        return wrapper
        
    @add_another
    def add_one(number): return number + 1
[/code]

So when we use a decorator, the code in the `wrapper` function is called instead of the original function. Typically the `wrapper` function calls the original function (otherwise there would be no point in decorating it as you'd just have a new unrelated function).

## Useful Example

For example, maybe you want to print (or log) particular function call times and the args. See this decorator that does just that (and can be used on methods too)

[code]

    from datetime import datetime
[/code]

[code]

    def print_decorator(func):
        def wrapper(*args, **kwargs):
            print(f"{datetime.now()}:{func}:args={args}:kwargs={kwargs}")
            return func(*args, **kwargs)
        return wrapper
[/code]

[code]

    @print_decorator
    def simple_add(a,b): return a + b
[/code]

[code]

    simple_add(2,4)
[/code]

[code]

    2022-11-02 14:18:56.635936:<function simple_add>:args=(2, 4):kwargs={}
    
[/code]

6

[code]

    @print_decorator
    def complex_add(a,b,*args,**kwargs): 
        out = a + b
        for arg in args: out = out + arg
        for kwarg in kwargs.values(): out = out + kwarg
        return out
[/code]

[code]

    complex_add(5,2,3,foo=6,bar=10)
[/code]

[code]

    2022-11-02 14:18:57.716085:<function complex_add>:args=(5, 2, 3):kwargs={'foo': 6, 'bar': 10}
    
[/code]

26

## Use on Existing Functions

What we have seen is applying a decorator to functions we fully define but we can also apply them to previously existing functions like ones we import from a library. This is helpful not just in understanding one way you can extend an existing libraries functionality, but also in understanding what decorators are. They aren't magical.

Let's add logging to `pd.DataFrame` using our existing decorator so we can see when a dataframe is constructed.

[code]

    LoggingDataFrame = print_decorator(pd.DataFrame)
    df = LoggingDataFrame([1,2,3])
[/code]

[code]

    2022-11-02 14:53:16.323144:<class 'pandas.core.frame.DataFrame'>:args=([1, 2, 3],):kwargs={}
    
[/code]

[code]

    df.head()
[/code]

| 0  
---|---  
0 | 1  
1 | 2  
2 | 3  
  
The key thing to notice here is that the `@` syntax really isn't doing anything magical. It's just passing the function into the decorator and using that as the function definition. It's just syntactic sugar for a higher order function that takes a function and returns a function.

To understand why this works, think through what our decorator is doing.

  1. It's a function that takes a function as an argument
  2. It creates a new function called `wrapper`. This `wrapper` function called the argument passed into it, but also has other code.
  3. It returns that function as the output

[code]

    print_function_source(print_decorator)
[/code]

[code]

    def print_decorator(func):
        def wrapper(*args, **kwargs):
            print(f"{datetime.now()}:{func}:args={args}:kwargs={kwargs}")
            return func(*args, **kwargs)
        return wrapper
    
[/code]

# Inheritance

Inheritance is the idea that you a class can "Inherit" attributes and methods from other classes.

For example a class could have an attribute `a`, and it can be used to create a new class to give it that attribute without having to specify it.

## Silly Simple Example

[code]

    class aClass: a = 2
        
    class bClass(aClass): pass
        
    aClass.a == bClass.a
[/code]

True

## Useful Examples

In many cases there are common things we want to inherit in lots of classes. One example is having access to the date. Often you want this for logging, or printing, or any number of things. By subclassing you don't have to reformat the date each time in your classes.

[code]

    class DateMinuteMixin:
        date_format='%Y%m%d_%H%M%S'
        dte = datetime.now()
    
        @property
        def date_str(self): return self.dte.strftime(self.date_format)
[/code]

Another handy use is to have generic behavior for handling different file types. In this case, we have a mixin where it opens and reads a sql file. Rather than rewriting this code for every class that needs to read a sql file, you can inherit from a class when you need that functionality.

> #### 💡 Tip
>
> You can define an abstract property like below to let users know that after inheriting this class, they need to define that property. In this case, they define the `sql_filepath`, and they get the contents of the file for free via the other methods.
[code]

    import abc
    
    class SqlFileMixin:
        @abc.abstractproperty
        def sql_filepath(self):
            pass
    
        @property
        def sql_file(self):
            return open(self.sql_filepath)
    
        @property
        def query(self):
            return self.sql_file.read()
[/code]

# Dunders

[code]

    import numpy as np
    class someClass:
        def __init__(self,a): self.a = a
        def __str__(self): return f"This object's a is : {self.a}"
        def __getitem__(self,idx): return self.a[idx-1]
        def __add__(self,some_class): return list(map(lambda x,y: x + y, self.a, some_class.a))
[/code]

[code]

    a = someClass(x)
    a.a
[/code]

[1, 2, 3, 4, 25]

[code]

    a + a
[/code]

[2, 4, 6, 8, 50]

[code]

    a[1]
[/code]

1

[code]

    a
[/code]

<__main__.someClass>

[code]

    print(a)
[/code]

[code]

    This object's a is : [1, 2, 3, 4, 25]
    
[/code]

# Iterators/Data Streaming

Iterators are useful when you don't want to just load all data in memory all at once. They are often defined with `yield`, but there are other ways.

## Silly Simple Example

[code]

    def mapper(items,fn):
        for item in items: yield item
[/code]

[code]

    it = mapper([2,4,6,8],square)
    it
[/code]

<generator object mapper>

[code]

    next(it), next(it), next(it)
[/code]

(2, 4, 6)

You can also process it sequentially in a loop.

[code]

    for item in mapper([2,4,6,8],square): 
        print(item)
[/code]

[code]

    2
    4
    6
    8
    
[/code]

## Useful Example

### File Streaming

[code]

    print_plus = partial(print,end='\n++++++\n')
    
    with open('test.txt', 'rb') as f:
        iterator = iter(partial(f.read, 64), b'')
        print_plus(type(iterator))
        for block in iterator: print_plus(block)
[/code]

[code]

    <class 'callable_iterator'>
    ++++++
    b'one\ntwo\nthree\nfour\nfive\nsix\nseven\neight\nnine\nten\neleven\ntwelve\nt'
    ++++++
    b'hirteen\nninety nine thousand nine hundred ninety\nninety nine tho'
    ++++++
    b'usand nine hundred ninety one\nninety nine thousand nine hundred '
    ++++++
    b'ninety two\nninety nine thousand nine hundred ninety three\nninety'
    ++++++
    b' nine thousand nine hundred ninety four\nninety nine thousand nin'
    ++++++
    b'e hundred ninety five\nninety nine thousand nine hundred ninety s'
    ++++++
    b'ix\nninety nine thousand nine hundred ninety seven\nninety nine th'
    ++++++
    b'ousand nine hundred ninety eight\nninety nine thousand nine hundr'
    ++++++
    b'ed ninety nine\n'
    ++++++
    
[/code]

### Stay Updated

Get notified about new posts on AI, web development, and tech insights.

Subscribe

## Contents

SetupPurposeComprehensionsDestructuringFastcoreParallel ProcessingDocmentsTesting LAttrDictLoggingHigher Order Functions & CallbacksFilter Map Simple LoggingFile ProcessorFormat and clean SQL fileDecoratorsSilly Simple ExampleUseful ExampleUse on Existing FunctionsInheritanceSilly Simple ExampleUseful ExamplesDundersIterators/Data StreamingSilly Simple ExampleUseful ExampleFile Streaming

## Let's Connect

GitHub

LinkedIn

Twitter

© 2025 Isaac Flath • All rights reserved

