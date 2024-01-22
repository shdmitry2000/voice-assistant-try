import traceback
import multiprocessing
import time

import asyncio


# openai.api_key = config("OPENAI_API_KEY")

def timing_decorator(func):
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds to execute.")
        return result
    return wrapper


def run_async_with_callback(callback):
    def inner(func):
        def wrapper(*args, **kwargs):
            def __exec():
                out = func(*args, **kwargs)
                callback(out)
                return out

            return asyncio.get_event_loop().run_in_executor(None, __exec)

        return wrapper

    return inner

def run_async():
    def inner(func):
        def wrapper(*args, **kwargs):
            def __exec():
                return func(*args, **kwargs)

            return asyncio.get_event_loop().run_in_executor(None, __exec)

        return wrapper

    return inner

def run_method(class_instance, method_name, *args, **kwargs):
        method = getattr(class_instance, method_name, None)
        if method is not None:
            return method(*args, **kwargs)
        else:
            print(f"No method named {method_name} found in the class")

