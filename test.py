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
    
    


import multiprocessing
import time

queue = multiprocessing.Queue()

def long_task(sleep_time):
   time.sleep(sleep_time)
   return sleep_time

def print_result(result):
   print(f"Task finished with sleep time: {result}")
   print("put in q",result)
   queue.put(result)
   
def main(): 
    pool = multiprocessing.Pool(processes=10) # limit to 10 processes
    tasks = []
    for _ in range(10): # replace 10 with the number of tasks you want to run
        sleep_time = int(input("Enter sleep time: "))
        task = pool.apply_async(long_task, (sleep_time,) )#, callback=print_result)
        tasks.append(task)
        task=tasks[0]
        if(task.ready()):
            print("task ready:",task.get(timeout=1)) # wait for the task to complete
            tasks.remove(task)

    
    for task in tasks:
        task.wait() # wait for the task to complete
        
    for task in tasks:
        print(task.get(timeout=2)) # wait for the task to complete
    
    # print(queue.empty())
    # while not queue.empty():
    #     print(f"Task finished with sleep time: {queue.get()}")

if __name__ == "__main__":
   main()