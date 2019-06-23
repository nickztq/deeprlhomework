import multiprocessing
import numpy as np

def worker(i, return_dict):
    # return_dict[i] = np.random.normal(size=5)
    return_dict.append(np.random.normal(size=5))

if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dict = manager.list()
    jobs = []
    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(i, return_dict))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()
    print(return_dict)
    # print (return_dict.values())