from scipy import signal, optimize
import numpy as np
import matplotlib.pyplot as plt

class Analyze:
    # 回帰直線のパラメータを求める関数
    def fitting(self, x, y):
        n = len(x)
        a = ((1 / n) * sum(x * y) - np.mean(x) * np.mean(y)) / ((1 / n) * sum(x ** 2) - (np.mean(x)) ** 2)
        b = np.mean(y) - a * np.mean(x)
        return a, b

    # Least squares method with scipy.optimize
    def fit_func(self, parameter, x, y):
        a = parameter[0]
        b = parameter[1]
        residual = y - (a * x + b)
        return residual

    def fit(self, x, y):

        a, b = np.polyfit(x, y, 1)
        # a, b = ID.fitting(x, y)

        # parameter0 = [0., 0.]
        # result = optimize.leastsq(Analyze.fit_func, parameter0, args=(x, y))
        # print(result)
        # a = result[0][0]
        # b = result[0][1]

        # a = 100
        # a = 0.035
        # a = 0.14

        # print(a, b)

        # x1 = np.arange(-200, 200, 0.1)
        # y1 = a * x + b
        # plt.plot(x1, y1)

        plt.scatter(y, x)
        xlim = 50
        ylim = 10
        plt.xlim(-xlim, xlim)
        plt.ylim(-ylim, ylim)
        plt.ylabel('current')
        plt.xlabel('Accel')
        plt.show()
        # plt.savefig("50%_pitch_sin_cos_3_if2=0.14.png")
