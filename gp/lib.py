import time

def tic():
    global tic_time
    tic_time = time.time()

def toc():
    return (time.time() - tic_time)
