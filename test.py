# # SuperFastPython.com
# # example of parallel map_async() with the process pool
# from random import random
# from time import sleep
# from multiprocessing.pool import Pool
 
# # task executed in a worker process
# def task(identifier):
#     # generate a value
#     value = random()
#     # report a message
#     print(f'Task {identifier} executing with {value}', flush=True)
#     # block for a moment
#     sleep(value)
#     # return the generated value
#     return value
 
# # protect the entry point
# if __name__ == '__main__':
#     # create and configure the process pool
#     pool = Pool(processes=10) # limit to 10 processes
#     for x in range(0,9):
#         # issues tasks to process pool
#         result = pool.map_async(task,( x,))
#         # iterate results
#         for result in result.get():
#             print(f'Got result: {result}', flush=True)
#     # process pool is closed automatically
    
    
#-----------------------------------------------------------------
import asyncio
import multiprocessing
import speech_recognition as sr
import voice,utility
from abc import ABC, abstractmethod

from audio_utility import *
class MultiprocessingHelper:
   def __init__(self,pool = multiprocessing.Pool(processes=2)):
       self.pool=pool

   async def perform_multiprocess(self, *args, **kwargs):
       loop = asyncio.get_event_loop()
       future = loop.run_in_executor(None, self.perform_task, *args, kwargs)
       return await future

   @abstractmethod
   def perform_task(self, *args, **kwargs):
        pass

class MultiprocessingTranscriberHelper(MultiprocessingHelper):
   def __init__(self,pool = multiprocessing.Pool(processes=2) ):
       super(MultiprocessingTranscriberHelper,self).__init__(pool)

    
   def perform_task(self, *args, **kwargs):
        r = sr.Recognizer()
        print(args)
        method_name= args[0]['method_name']   # Accessing 'method_name'
        audio_data = args[0]['audio_data'] # Accessing audio_data
        language = args[0]['language'] # Accessing language
        
        try:
            print(method_name)
            text=utility.run_method(r,method_name,audio_data=audio_data,language=language)
            # text = r.recognize_google(audio_data=audio_data,language=language)
            print(text)
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return "Could not request results from Google Speech Recognition service; {0}".format(e)

class AsyncClass:
 def __init__(self):
   self.helper = MultiprocessingTranscriberHelper()

 async def perform_multi_process(self, *args, **kwarg):
   return await self.helper.perform_multiprocess(*args, **kwarg)

# Usage
async def main():
   async_class = AsyncClass()
   filename='/Users/dmitryshlymovich/workspace/wisper/voice-assistant-chatgpt/userQuestion_k7xJngMTUo.wav'
   audio_data=load_audioSource_from_file(filename) 
     
   result = await async_class.perform_multi_process(method_name="recognize_google", audio_data=audio_data, language="he")
   print(result)

asyncio.run(main())





#---------------------------------------------------------------

# import multiprocessing
# import time

# import asyncio
# import requests


# def run_async_with_callback(callback):
#     def inner(func):
#         def wrapper(*args, **kwargs):
#             def __exec():
#                 out = func(*args, **kwargs)
#                 callback(out)
#                 return out

#             return asyncio.get_event_loop().run_in_executor(None, __exec)

#         return wrapper

#     return inner

# def run_async():
#     def inner(func):
#         def wrapper(*args, **kwargs):
#             def __exec():
#                 return func(*args, **kwargs)

#             return asyncio.get_event_loop().run_in_executor(None, __exec)

#         return wrapper

#     return inner

# def _callback(*args):
#     # asyncio.wait(1)
#     print(args)


# # Must provide a callback function, callback func will be executed after the func completes execution !!
# @run_async_with_callback(_callback)
# # @run_async
# def get(url):
#     out= requests.get(url)
#     # task = asyncio.create_task(_callback(out))
#     # await task
#     print("get end")
#     return out
    


# get("https://google.com")
# print("Non blocking code ran !!")


#------------------------------------------------------------------------

# import asyncio

# async def first_function():
#     print("First function started")
#     task =asyncio.create_task(second_function())
#     # Continue with the rest of the first function
#     print("First function continued")
#     # Optionally, you can wait for the second function to finish here
#     #  await task
#     print("First function finished")

# async def second_function():
#     print("Second function started")
#     # await asyncio.sleep(1) # Simulate work
#     print("Second function finished")

# asyncio.run(first_function())
# time.sleep(2)
# print("end")
#-----------------------------------------------------------------

# async def main():
#     try:
        
#         async with asyncio.TaskGroup() as tg:
#             tg.create_task(get("https://google.com"))
                
            
#             print("Non blocking code ran !!")       
#             # done, pending = await asyncio.wait([sh.add_chank(chank), print(await sh.get_current_text()),sh.finalize_data()])
#             # done, pending = await asyncio.wait([alive_task])

#             # for task in pending:
            
#                 # task.cancel()
#     except (asyncio.exceptions.CancelledError,TypeError) as e:
#             print("task cancel ,error",e)
        
        
# asyncio.run(main())   


#--------------------------------------------

# def long_task(sleep_time):
#    time.sleep(sleep_time)
#    return sleep_time

# def print_result(result):
#    print(f"Task finished with sleep time: {result}")
#    print("put in q",result)
#    queue.put(result)
   
# def main(): 
#     pool = multiprocessing.Pool(processes=10) # limit to 10 processes
#     tasks = []
#     for _ in range(10): # replace 10 with the number of tasks you want to run
#         sleep_time = int(input("Enter sleep time: "))
#         task = pool.apply_async(long_task, (sleep_time,) )#, callback=print_result)
#         tasks.append(task)
#         task=tasks[0]
#         if(task.ready()):
#             print("task ready:",task.get(timeout=1)) # wait for the task to complete
#             tasks.remove(task)

    
#     for task in tasks:
#         task.wait() # wait for the task to complete
        
#     for task in tasks:
#         print(task.get(timeout=2)) # wait for the task to complete
    
#     # print(queue.empty())
#     # while not queue.empty():
#     #     print(f"Task finished with sleep time: {queue.get()}")

# if __name__ == "__main__":
#    main()