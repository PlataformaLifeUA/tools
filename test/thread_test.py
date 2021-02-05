from concurrent.futures.thread import ThreadPoolExecutor
from os import cpu_count
from random import randint
from time import sleep


def f(arg1: int, arg2: int):
    t = randint(0, arg1 * arg2)
    print(f'Waitting {t} seconds')
    sleep(t)
    print(f'Fisnishing to wait {t} seconds')
    return t


with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
    futures = []
    for i in range(100):
        futures.append(executor.submit(f, i, 2))
    for future in futures:
        print(future.result())

