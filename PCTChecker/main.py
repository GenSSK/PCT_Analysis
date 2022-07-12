from __future__ import print_function
from socket import *
import time
from contextlib import closing
import Npz


def main():
    host = '192.168.2.30'
    host = gethostbyname(host)
    print(host)
    port = 5555
    count = 0
    sock = socket(AF_INET, SOCK_DGRAM)

    npz = Npz.NPZ()
    data = npz.single_load('/cfo/2022-07-01_y.inoue_k.tozuka_b.poitorimol_y.baba_1234_CFO.npz')
    # data = npz.single_load('npz/cooperation/2022-07-01_y.inoue_k.tozuka_b.poitorimol_y.baba_1234.npz')

    send_string = []
    message = []

    dec = 1000

    for i in range(len(data['time'][::dec])):
        send_string.append(str(data['pitch'][i * dec]) + ',' + \
                           str(data['roll'][i * dec]) + ',' + \
                           str(-data['targety'][i * dec]) + ',' + \
                           str(data['targetx'][i * dec]) + ',' + \
                           str(-data['bally_pre'][i * dec]) + ',' + \
                           str(data['ballx_pre'][i * dec]) + ',' + \
                           str(data['task_angle'][i * dec]) + ',' + \
                           str(0.03) + ',' + \
                           str(2)
                           )
        message.append('{0}'.format(send_string[i]).encode('utf-8'))

    with closing(sock):
        for i in range(len(data['time'][::dec])):
            # print(send_string)

            # print(message)
            sock.sendto(message[i], (host, port))
            # count += 1
            time.sleep(0.0001 * float(dec))
    return


if __name__ == '__main__':
    main()
