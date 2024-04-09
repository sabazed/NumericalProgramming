import time


class Benchmark:

    def __init__(self):
        self.__bench = dict()

    def start_bench(self, name):
        stamp = time.time()
        self.__bench[name] = stamp
        print(f"Starting timestamp for {name}: {stamp}")

    def end_bench(self, name):
        stamp = time.time()
        print(f"Ending timestamp for {name}: {stamp}")
        print(f"Total running duration of {name}: {stamp - self.__bench[name]}")
        del self.__bench[name]
