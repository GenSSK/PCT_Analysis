import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft


def low_pass_filter(signal, smp, cutoff):
    signal_filtered = stack('low_pass', signal, smp, cutoff_low=cutoff)
    return signal_filtered


def high_pass_filter(signal, smp, cutoff):
    signal_filtered = stack('high_pass', signal, smp, cutoff_high=cutoff)
    return signal_filtered


def band_pass_filter(signal, smp, cutoff_low, cutoff_high):
    signal_filtered = stack('band_pass', signal, smp, cutoff_low=cutoff_low, cutoff_high=cutoff_high)
    return signal_filtered


def band_eliminate_filter(signal, smp, cutoff_low, cutoff_high):
    signal_filtered = stack('band_eliminate', signal, smp, cutoff_low=cutoff_low, cutoff_high=cutoff_high)
    return signal_filtered


def stack(mode: str, signal, smp, cutoff_low=0, cutoff_high=0):
    signal_flatten = signal.reshape(-1, signal.shape[-1])
    sig_ = []
    for sig in signal_flatten:
        sig_.append(calc_filter(mode, sig, smp, cutoff_low, cutoff_high))
    sig_array = np.array([_ for _ in sig_])
    signal_filtered = sig_array.reshape(signal.shape)

    return signal_filtered


def calc_filter(mode: str, signal, smp, cutoff_low=0, cutoff_high=0):
    spect = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=smp)
    if mode == 'low_pass':
        spect[np.abs(freq) > cutoff_low] = 0
    if mode == 'high_pass':
        spect[np.abs(freq) < cutoff_high] = 0
    if mode == 'band_pass':
        if cutoff_low > cutoff_high:
            cutoff_low, cutoff_high = cutoff_high, cutoff_low
        spect[(np.abs(freq) > cutoff_high) | (np.abs(freq) < cutoff_low)] = 0
    if mode == 'band_eliminate':
        if cutoff_low > cutoff_high:
            cutoff_low, cutoff_high = cutoff_high, cutoff_low
        spect[(np.abs(freq) < cutoff_high) & (np.abs(freq) > cutoff_low)] = 0

    signal_filtered = np.fft.ifft(spect)
    signal_filtered = signal_filtered.real.astype(np.float32)

    return signal_filtered


# test sample
if __name__ == '__main__':
    smp = 0.001
    time = np.arange(0, 5, smp)
    data = np.random.randn(len(time))

    # フィルターの設定
    cutoff_low = 200
    cutoff_high = 300

    # フィルターをかける
    data_low_pass = low_pass_filter(data, smp, cutoff_low)
    data_high_pass = high_pass_filter(data, smp, cutoff_high)
    data_band_pass = band_pass_filter(data, smp, cutoff_low, cutoff_high)
    data_band_eliminate = band_eliminate_filter(data, smp, cutoff_low, cutoff_high)

    # 以下、グラフ解析用
    data_list = [
        data, data_low_pass, data_high_pass, data_band_pass, data_band_eliminate
    ]
    data_name_list = [
        'Raw', 'LowPass', 'HighPass', 'BandPass', 'BandEliminate'
    ]

    fig, ax = plt.subplots(6, 1, figsize=(10, 15))
    for i, sig in enumerate(data_list):
        spect = np.fft.fft(sig)
        freq = np.fft.fftfreq(len(sig), d=smp)
        Amp = np.abs(spect / (len(sig) / 2))   # 振幅
        ax[i].plot(freq[1:int(len(sig) / 2)], Amp[1:int(len(sig) / 2)], label=data_name_list[i])
        ax[i].set_title(data_name_list[i])
        ax[i].set_xlabel('Frequency (Hz)')
        ax[i].set_ylabel('Amplitude')
        ax[5].plot(time, sig, label=data_name_list[i])
        ax[5].set_title('Time domain')
        ax[5].set_xlabel('Time (s)')
        ax[5].set_ylabel('Amplitude')
        ax[5].legend()
    plt.tight_layout()
    plt.show()