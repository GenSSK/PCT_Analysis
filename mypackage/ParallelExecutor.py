import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

class ParallelExecutor:
    def __init__(self, func, max_workers=os.cpu_count(), args_num: int = 1, return_num = 1, para_args_list=None, Executor='Thread'):
        if para_args_list is None:
            para_args_list = []
        self.max_workers = max_workers
        self.args_num = args_num
        self.return_num = return_num
        self.para_args_list = para_args_list
        self.func = func
        self.Executor = Executor

    def parallel(self, *args):
        if self.Executor == 'Thread':
            tpe = ThreadPoolExecutor(max_workers=self.max_workers)
        elif self.Executor == 'Process':
            tpe = ProcessPoolExecutor(max_workers=self.max_workers)
        para = []
        dim_is_one = False
        for i in range(len(self.para_args_list)):
            if args[self.para_args_list[i]].ndim == 1:
                flatten = args[self.para_args_list[i]]
                dim_is_one = True
            else:
                flatten = args[self.para_args_list[i]].reshape(-1, args[self.para_args_list[i]].shape[-1])
            para.append(flatten)

        results_ = []
        for i in range(len(para[0])):
            args_ = self._make_args(para, i, *args)
            fix_args = (_ for _ in args_)
            results_.append([])
            res = tpe.submit(self.func, *fix_args).result()
            results_[i] = res

        tpe.shutdown()

        # print(results_)
        if dim_is_one:
            if self.return_num == 1:
                ret = np.array(results_)
            else:
                results = np.array(results_)
                results = results.T
                ret = results
        else:
            if self.return_num == 1:
                results = np.array(results_)
                # print(args[self.para_args_list[0]].shape[:-1] + res.shape)
                ret = results.reshape(args[self.para_args_list[0]].shape[:-1] + res.shape)
            else:
                results = []
                for i in range(self.return_num):
                    results.append([])
                    for j in range(len(results_)):
                        results[i].append(results_[j][i])
                for i in range(self.return_num):
                    results[i] = np.array(results[i]).reshape(args[self.para_args_list[0]].shape[:-1] + results[i][0].shape)
                ret = (_ for _ in results)

        return ret


    def _make_args(self, para, para_num, *args):
        args_ = []
        para_count = 0
        for i in range(len(args)):
            if i in self.para_args_list:
                args_.append(para[para_count][para_num])
                para_count += 1
            else:
                args_.append(args[i])
        return args_

if __name__ == "__main__":
    def test1(a, b, c):
        ans1 = a + b * c
        return ans1
    def test2(a, b, c):
        ans1 = (a + b) * c
        ans2 = a / a
        return ans1, ans2


    a1 = np.arange(1, 100, 0.001)
    b1 = np.arange(101, 201, 0.001)
    c1 = 3

    a2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b2 = np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]])
    c2 = 3

    # Normal calculation
    # test1 1D
    start = time.time()
    normal_ans_1_1d = np.zeros(len(a1))
    for i in range(len(a1)):
        normal_ans_1_1d[i] = test1(a1[i], b1[i], c1)
    normal_test1_1d_time = time.time() - start


    # test1 2D
    start = time.time()
    normal_ans_1_2d = np.zeros(a2.shape)
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            normal_ans_1_2d[i, j] = test1(a2[i, j], b2[i, j], c2)
    normal_test1_2d_time = time.time() - start

    # test2 1D
    start = time.time()
    normal_ans_2_1d_1 = np.zeros(len(a1))
    normal_ans_2_1d_2 = np.zeros(len(a1))
    for i in range(len(a1)):
        normal_ans_2_1d_1[i], normal_ans_2_1d_2[i] = test2(a1[i], b1[i], c1)
    normal_test2_1d_time = time.time() - start

    # test2 2D
    start = time.time()
    normal_ans_2_2d_1 = np.zeros(a2.shape)
    normal_ans_2_2d_2 = np.zeros(a2.shape)
    for i in range(a2.shape[0]):
        for j in range(a2.shape[1]):
            normal_ans_2_2d_1[i, j], normal_ans_2_2d_2[i, j] = test2(a2[i, j], b2[i, j], c2)
    normal_test2_2d_time = time.time() - start

    # Parallel calculation
    # test1 1D
    start = time.time()
    pe = ParallelExecutor(test1, max_workers=12, args_num=3, return_num=1, para_args_list=[0, 1])
    parallel_ans_1_1d = pe.parallel(a1, b1, c1)
    parallel_test1_1d_time = time.time() - start

    # test1 2D
    start = time.time()
    pe = ParallelExecutor(test1, max_workers=12, args_num=3, return_num=1, para_args_list=[0, 1])
    parallel_ans_1_2d = pe.parallel(a2, b2, c2)
    parallel_test1_2d_time = time.time() - start

    # test2 1D
    start = time.time()
    pe = ParallelExecutor(test2, max_workers=12, args_num=3, return_num=2, para_args_list=[0, 1])
    parallel_ans_2_1d_1, parallel_ans_2_1d_2 = pe.parallel(a1, b1, c1)
    parallel_test2_1d_time = time.time() - start

    # test2 2D
    start = time.time()
    pe = ParallelExecutor(test2, max_workers=12, args_num=3, return_num=2, para_args_list=[0, 1])
    parallel_ans_2_2d_1, parallel_ans_2_2d_2 = pe.parallel(a2, b2, c2)
    parallel_test2_2d_time = time.time() - start

    print("Answer check")
    print("test1 1D -> ", np.allclose(normal_ans_1_1d, parallel_ans_1_1d))
    print("test1 2D -> ", np.allclose(normal_ans_1_2d, parallel_ans_1_2d))
    print("test2 1D -> ", np.allclose(normal_ans_2_1d_1, parallel_ans_2_1d_1), np.allclose(normal_ans_2_1d_2, parallel_ans_2_1d_2))
    print("test2 2D -> ", np.allclose(normal_ans_2_2d_1, parallel_ans_2_2d_1), np.allclose(normal_ans_2_2d_2, parallel_ans_2_2d_2))

    print("Time check   Normal vs. Parallel")
    print("test1 1D -> ", normal_test1_1d_time, parallel_test1_1d_time)
    print("test1 2D -> ", normal_test2_2d_time, parallel_test1_2d_time)
    print("test2 1D -> ", normal_test2_1d_time, parallel_test2_1d_time)
    print("test2 2D -> ", normal_test2_2d_time, parallel_test2_2d_time)