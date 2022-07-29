from __future__ import print_function
from socket import *
import time
from contextlib import closing

import numpy as np

import Npz
import pickle

def main():
    host = '192.168.2.30'
    host = gethostbyname(host)
    port = 5555
    count = 0
    sock = socket(AF_INET, SOCK_DGRAM)

    npz = Npz.NPZ()
    data = npz.single_load('/cfo/2022-07-01_y.inoue_k.tozuka_b.poitrimol_y.baba_1234_CFO.npz')
    # data = npz.single_load('npz/cooperation/2022-07-01_y.inoue_k.tozuka_b.poitorimol_y.baba_1234.npz')
    # data = npz.single_load('npz/2022-07-21_g.sasaki_1.npz')
    # data = npz.single_load('npz/2022-07-01_y.inoue_1_trans.npz')

    send_string = []
    message = []

    dec = 1000

    reading_file = []
    filename = 'tes'

    overwrite = True

    if overwrite:
        print('make file')
        for i in range(len(data['time'][::dec])):
            send_string.append(str(data['pitch'][i * dec]) + ',' + \
                               str(data['roll'][i * dec]) + ',' + \
                               str(data['targetx'][i * dec]) + ',' + \
                               str(data['targety'][i * dec]) + ',' + \
                               str(data['ballx'][i * dec]) + ',' + \
                               str(data['bally'][i * dec]) + ',' + \
                               str(data['task_angle'][i * dec]) + ',' + \
                               str(data['i1_cfo_magnitude'][i * dec]) + ',' + \
                               str(data['i1_cfo_angle'][i * dec]) + ',' + \
                               str(data['i1_ecfo'][i * dec]) + ',' + \
                               str(data['i1_inecfo'][i * dec]) + ',' + \
                               str(0.03) + ',' + \
                               str(2)
                               )
            message.append('{0}'.format(send_string[i]).encode('utf-8'))

            with open(filename, 'wb') as w:
                pickle.dump(message, w)

    try:
        reading_file = open(filename, 'rb')
    except OSError:
        for i in range(len(data['time'][::dec])):
            print('make file')
            send_string.append(str(data['i2_p_thm'][i * dec]) + ',' + \
                               str(data['i2_r_thm'][i * dec]) + ',' + \
                               str(data['targetx'][i * dec]) + ',' + \
                               str(data['targety'][i * dec]) + ',' + \
                               str(data['ballx'][i * dec]) + ',' + \
                               str(data['bally'][i * dec]) + ',' + \
                               str(data['task_angle'][i * dec]) + ',' + \
                               str(data['i1_cfo_magnitude'][i * dec]) + ',' + \
                               str(data['i1_cfo_angle'][i * dec]) + ',' + \
                               str(data['i1_ecfo'][i * dec]) + ',' + \
                               str(data['i1_inecfo'][i * dec]) + ',' + \
                               str(0.03) + ',' + \
                               str(2)
                               )
            message.append('{0}'.format(send_string[i]).encode('utf-8'))

            with open(filename, 'wb') as w:
                pickle.dump(message, w)

    else:
        message = pickle.load(reading_file)
        print('file read')

    with closing(sock):
        print('send')
        for i in range(len(data['time'][::dec])):
            sock.sendto(message[i], (host, port))
            time.sleep(0.0001 * float(dec))
    return


if __name__ == '__main__':
    main()
