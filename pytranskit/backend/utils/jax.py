import numpy as np
from skimage.transform import radon, iradon
from joblib import Parallel, delayed #Used for RadonCDT, comes with skimage

import jax
import jax.numpy as jnp
import functools # For jax.jit static methods
from typing import NamedTuple, Tuple, Any, Callable
import math



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                          jax configuration for gpu/cpu targeting
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_pipeline_hardware(target_hardware='gpu'):
    """  Sets global backend execution target ('gpu' or 'cpu')   \n
        Try to avoid setting this multiple times"""
    try:
        if (target_hardware is None): return
        jax.config.update('jax_platform_name', target_hardware.lower())
        jnp.linspace(0,1,10) #Force jax to make a computation to check it
        print(f"Pipeline target successfully configured for: {target_hardware.upper()}")
    except RuntimeError as e:
        print(f"Failed setting hardware backend choice: {e}")





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                               jax.jit utility methods
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def jit_func(*args, **kwargs):
    """Wrap a function with jit and optional args. (Used to keep docstrings in editor)"""
    def wrapper(func): return functools.wraps(func)(jax.jit(func, *args, **kwargs))
    return wrapper

def jit_static(*args, **kwargs):
    """Jit a function, preserve types, and make it a staticmethod. """
    return staticmethod(jit_func)
