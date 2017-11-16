import time

def tic():
    global tic_time
    tic_time = time.time()

def toc():
    return (time.time() - tic_time)



def is_number(str):
    try:
        complex(str) # for int, long, float and complex
    except ValueError:
        return False

    return True