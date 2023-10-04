import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter


def hilbert(sig, dt):
    if sig.ndim == 1:
        env, freq_inst, phase_inst = calc_hilbert(sig, dt)
    if sig.ndim == 2:
        env = np.zeros(sig.shape)
        freq_inst = np.zeros(sig.shape)
        phase_inst = np.zeros(sig.shape)
        for i in range(sig.shape[0]):
            env[i, :], freq_inst[i, :], phase_inst[i, :] = calc_hilbert(sig[i, :], dt)
    if sig.ndim == 3:
        env = np.zeros(sig.shape)
        freq_inst = np.zeros(sig.shape)
        phase_inst = np.zeros(sig.shape)
        for i in range(sig.shape[0]):
            for j in range(sig.shape[1]):
                env[i, j, :], freq_inst[i, j, :], phase_inst[i, j, :] = calc_hilbert(sig[i, j, :], dt)
    if sig.ndim == 4:
        env = np.zeros(sig.shape)
        freq_inst = np.zeros(sig.shape)
        phase_inst = np.zeros(sig.shape)
        for i in range(sig.shape[0]):
            for j in range(sig.shape[1]):
                for k in range(sig.shape[2]):
                    env[i, j, k, :], freq_inst[i, j, k, :], phase_inst[i, j, k, :] = calc_hilbert(sig[i, j, k, :], dt)

    if sig.ndim == 5:
        env = np.zeros(sig.shape)
        freq_inst = np.zeros(sig.shape)
        phase_inst = np.zeros(sig.shape)
        for i in range(sig.shape[0]):
            for j in range(sig.shape[1]):
                for k in range(sig.shape[2]):
                    for l in range(sig.shape[3]):
                        env[i, j, k, l, :], freq_inst[i, j, k, l, :], phase_inst[i, j, k, l, :] = calc_hilbert(sig[i, j, k, l, :], dt)

    return env, freq_inst, phase_inst


def calc_hilbert(sig, dt):
    # ヒルベルト変換し，包絡線と瞬時周波数を得る
    jsgn = np.sign(np.fft.fftfreq(len(sig), dt)) * 1.0j  # 1/(pi*t)のフーリエ変換のマイナス
    hsig = np.fft.ifft(-jsgn * np.fft.fft(sig))  # フーリエ空間で積を取りフーリ逆変換 (畳み込み積分の計算)
    env = np.sqrt(sig ** 2 + hsig ** 2)  # 包絡線（エンベロープ）
    phase_inst = np.arctan2(np.real(hsig), sig)  # 瞬時位相
    freq_inst = np.gradient(np.unwrap(phase_inst)) / dt / (2.0 * np.pi)  # 瞬時周波数

    # # scipy を使用するとき
    # z = signal.hilbert(sig)
    # phase_inst = np.unwrap(np.angle(z))
    # freq_inst = np.gradient(phase_inst)/dt/(2.0*np.pi)
    # #

    return env.real, freq_inst, phase_inst


def relative_phase(phase1, phase2, sigma: int = 'none'):
    if phase1.ndim == 1:
        rela_phase = calc_relative_phase(phase1, phase2, sigma)
    if phase1.ndim == 2:
        rela_phase = np.zeros(phase1.shape)
        for i in range(phase1.shape[0]):
            rela_phase[i, :] = calc_relative_phase(phase1[i, :], phase2[i, :], sigma)
    if phase1.ndim == 3:
        rela_phase = np.zeros(phase1.shape)
        for i in range(phase1.shape[0]):
            for j in range(phase1.shape[1]):
                rela_phase[i, j, :] = calc_relative_phase(phase1[i, j, :], phase2[i, j, :], sigma)
    if phase1.ndim == 4:
        rela_phase = np.zeros(phase1.shape)
        for i in range(phase1.shape[0]):
            for j in range(phase1.shape[1]):
                for k in range(phase1.shape[2]):
                    rela_phase[i, j, k, :] = calc_relative_phase(phase1[i, j, k, :], phase2[i, j, k, :], sigma)
    return rela_phase


def calc_relative_phase(phase1, phase2, sigma: int = 'none'):
    phase1 = phase1 * 180 / np.pi  # rad -> deg
    phase2 = phase2 * 180 / np.pi  # rad -> deg
    rela_phase = np.subtract(phase1, phase2)  # 差分を計算
    rela_phase = np.abs(rela_phase)  # 絶対値を計算
    rela_phase = np.where(rela_phase > 180, 360 - rela_phase, rela_phase)  # 180度以上のときは折り返す
    if sigma == 'none':
        return rela_phase
    else:
        rela_phase_smthd = gaussian_filter(rela_phase, sigma=sigma)
        return rela_phase_smthd


if __name__ == "__main__":
    # テスト信号の生成
    dt = 1.0e-4  # 時間刻み
    tms = 0.0  # 初期時間
    tme = 2.0  # 終了時間
    tm = np.arange(tms, tme, dt)  # 時間のnumpy配列

    amp = abs(np.sin(2.0 * np.pi * 2.0 * tm)) + np.tanh(tm)  # 振幅 (これが包絡線に相当する)
    freq = 10.0 + 10.0 * tm * tm + np.cos(2.0 * np.pi * 3.0 * tm)  # 周波数 (注意！これは瞬時周波数とは一致しません）
    sig = amp * np.sin(2.0 * np.pi * freq * tm)  # テスト信号

    env, freq_inst, phase_inst = hilbert(sig, dt)

    # プロット
    fig, (ax01, ax02) = plt.subplots(nrows=2, figsize=(6, 8))
    plt.subplots_adjust(wspace=0.0, hspace=0.6)

    # 入力信号と包絡線のプロット
    ax01.set_xlim(tms, tme)
    ax01.set_xlabel('time (s)')
    ax01.set_ylabel('x')
    ax01.plot(tm, sig, color='blue')  # 入力信号
    ax01.plot(tm, env, color='red', linestyle='dashed')

    # スペクトログラムと瞬時周波数
    ax02.set_xlim(tms, tme)
    ax02.set_ylim(0.0, 300.0)
    ax02.set_xlabel('time (s)')
    ax02.plot(tm, freq_inst, color='red', linestyle='dashed')

    # 2信号間の位相差を計算してプロット
    smp = 0.0001
    time = np.arange(20, 80, smp)
    sig1 = np.zeros(len(time))
    sig1[time < 30] += np.sin(2.0 * np.pi * 1.0 * time[time < 30])
    sig1[(30 < time) & (time < 40)] += np.sin(
        2.0 * np.pi * 1.0 * time[(30 < time) & (time < 40)] + (90 / 360) * 2 * np.pi)
    sig1[(40 < time) & (time < 50)] += np.sin(
        2.0 * np.pi * 1.0 * time[(40 < time) & (time < 50)] + (180 / 360) * 2 * np.pi)
    sig1[(50 < time) & (time < 60)] += np.sin(
        2.0 * np.pi * 1.0 * time[(50 < time) & (time < 60)] + (270 / 360) * 2 * np.pi)
    sig1[(60 < time) & (time < 70)] += np.sin(
        2.0 * np.pi * 1.0 * time[(60 < time) & (time < 70)] + (360 / 360) * 2 * np.pi)
    sig1[(70 < time) & (time < 80)] += np.sin(
        2.0 * np.pi * 1.0 * time[(70 < time) & (time < 80)] + (-90 / 360) * 2 * np.pi)
    sig2 = np.sin(2.0 * np.pi * 1.0 * time)
    env1, freq_int1, phase_inst1 = hilbert(sig1, smp)
    env2, freq_int2, phase_inst2 = hilbert(sig2, smp)
    rela_phase = relative_phase(phase_inst1, phase_inst2, sigma=300)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    plt.xlabel('time (s)')
    ax[0].plot(time, sig1)
    ax[0].plot(time, sig2)
    ax[0].set_ylabel('Signal')
    ax[1].plot(time, rela_phase)
    ax[1].set_ylabel('Relative phase (deg)')
    ax[1].set_ylim(0, 180)
    ax[1].set_yticks([0, 90, 180])

    plt.show()
