import pandas as pd
import glob

class readCSV:
    def __init__(self):

    def csv_read(self):
        # CSV読み込むよ！
        self.data = pd.read_csv('csv/DATA.csv')