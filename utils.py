import os
import shutil
import functools
import time
import inspect
import operator

import torch

def timeit(function):
    @functools.wraps(function)
    def timed(*args, **kwargs):
        start = time.time()
        result = function(*args, **kwargs)
        end = time.time()
        print(f'function name: {function.__name__}, args: {args}, kwargs: {kwargs}')
        print(f'Call took {end-start:.6f} seconds\n')
        return result
    return timed

def debug(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        import pdb; pdb.set_trace()
        return function(*args, **kwargs)
    return wrapped

def lineinfo(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        print(f'filename: {inspect.stack()[1][1]}',
              f'line number: {inspect.stack()[1][2]}',
              f'function name: {function.__name__}') 
        return function(*args, **kwargs)
    return wrapped

def safe_copytree(src, tgt):
    if not os.path.exists(src):
        raise OSError(f'Path to remove {tgt} does not exist')

    if os.path.exists(tgt):
        print(f'Target directory {tgt} already exists')
        return

    try:
        shutil.copytree(src, tgt)

    except OSError as e:
        if e.errno == errno.ENOSPC:
            logging.info('Copying failed (no space on disk).')
            logging.info(f'Removing {tgt}')
            if os.path.exists(tgt):
                shutil.rmtree(tgt)
            raise

def safe_rmtree(tgt):
    if not os.path.exists(tgt):
        raise OSError(f'Path to remove {tgt} does not exist')

    shutil.rmtree(tgt)

def param_size(module):
    '''
    computes memory use in MB of parameters of PyTorch module
    '''
    params = module.parameters(recurse=True)
    mb = 0
    for param in params:
        mb += param.element_size() * functools.reduce(operator.mul, param.shape)
    return mb/1e6
