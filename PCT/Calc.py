import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt


class Calculation:
    def __init__(self, data):
        self.data = data;

    def performance_calc(self, data):
        self.error = np.sqrt(
            (data['targetx'] - data['ballx']) ** 2 + (data['targety'] - data['bally']) ** 2)

        plt.plot(data['time'], self.error)
        plt.show()

    def period_performance(self):
        period = self.data['during_time']
