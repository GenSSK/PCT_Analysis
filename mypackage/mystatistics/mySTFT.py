import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patheffects as path_effects


def stft(signal, time, t_wndw=100.0e-3, n_stft=100, wndw='hamming'):
    tm_sp_, freq_sp_, sp_ = [], [], []
    signal_flatten = signal.reshape(-1, signal.shape[-1])
    t = time
    if signal.shape != time.shape:
        t = np.array([time for i in range(len(signal_flatten))])
    for i in range(len(signal_flatten)):
        tm_sp_ret, freq_sp_ret, sp_ret = calc_stft(signal_flatten[i], t[i], t_wndw=t_wndw, n_stft=n_stft, wndw=wndw)
        tm_sp_.append(tm_sp_ret)
        freq_sp_.append(freq_sp_ret)
        sp_.append(sp_ret)
    tm_sp_array = np.array([_ for _ in tm_sp_])
    freq_sp_array = np.array([_ for _ in freq_sp_])
    sp_array = np.array([_ for _ in sp_])
    tm_sp = tm_sp_array.reshape(signal.shape[:-1] + (-1,))
    freq_sp = freq_sp_array.reshape(signal.shape[:-1] + (-1,))
    sp = sp_array.reshape(signal.shape[:-1] + (sp_ret.shape))
    #
    # if signal.ndim == 1:
    #     tm_sp, freq_sp, sp = calc_stft(signal, time, t_wndw=t_wndw, n_stft=n_stft, wndw=wndw)
    # if signal.ndim == 2:
    #     tm_sp_ = []
    #     freq_sp_ = []
    #     sp_ = []
    #     for i in range(signal.shape[0]):
    #         tm_sp_ret, freq_sp_ret, sp_ret = calc_stft(signal[i, :], time, t_wndw=t_wndw, n_stft=n_stft, wndw=wndw)
    #         tm_sp_.append(tm_sp_ret)
    #         freq_sp_.append(freq_sp_ret)
    #         sp_.append(sp_ret)
    #     tm_sp = np.array([_ for _ in tm_sp_])
    #     freq_sp = np.array([_ for _ in freq_sp_])
    #     sp = np.array([_ for _ in sp_])
    #
    #     tm_sp = tm_sp.reshape([signal.shape[0], -1])
    #     freq_sp = freq_sp.reshape([signal.shape[0], -1])
    #     sp = sp.reshape([signal.shape[0], sp_ret.shape[0], -1])
    #
    #     return tm_sp, freq_sp, sp
    #
    # if signal.ndim == 3:
    #     tm_sp_ = []
    #     freq_sp_ = []
    #     sp_ = []
    #     for i in range(signal.shape[0]):
    #         for j in range(signal.shape[1]):
    #             tm_sp_ret, freq_sp_ret, sp_ret = calc_stft(signal[i, j, :], time, t_wndw=t_wndw, n_stft=n_stft, wndw=wndw)
    #             tm_sp_.append(tm_sp_ret)
    #             freq_sp_.append(freq_sp_ret)
    #             sp_.append(sp_ret)
    #     tm_sp = np.array([_ for _ in tm_sp_])
    #     freq_sp = np.array([_ for _ in freq_sp_])
    #     sp = np.array([_ for _ in sp_])
    #
    #     tm_sp = tm_sp.reshape([signal.shape[0], signal.shape[1], -1])
    #     freq_sp = freq_sp.reshape([signal.shape[0], signal.shape[1], -1])
    #     sp = sp.reshape([signal.shape[0], signal.shape[1], sp_ret.shape[0], -1])
    #
    # if signal.ndim == 4:
    #     tm_sp_ = []
    #     freq_sp_ = []
    #     sp_ = []
    #     for i in range(signal.shape[0]):
    #         for j in range(signal.shape[1]):
    #             for k in range(signal.shape[2]):
    #                 tm_sp_ret, freq_sp_ret, sp_ret = calc_stft(signal[i, j, k, :], time, t_wndw=t_wndw, n_stft=n_stft, wndw=wndw)
    #                 tm_sp_.append(tm_sp_ret)
    #                 freq_sp_.append(freq_sp_ret)
    #                 sp_.append(sp_ret)
    #     tm_sp = np.array([_ for _ in tm_sp_])
    #     freq_sp = np.array([_ for _ in freq_sp_])
    #     sp = np.array([_ for _ in sp_])
    #
    #     tm_sp = tm_sp.reshape([signal.shape[0], signal.shape[1], signal.shape[2], -1])
    #     freq_sp = freq_sp.reshape([signal.shape[0], signal.shape[1], signal.shape[2], -1])
    #     sp = sp.reshape([signal.shape[0], signal.shape[1], signal.shape[2], sp_ret.shape[0], -1])

    return tm_sp, freq_sp, sp


def calc_stft(signal, time, t_wndw=100.0e-3, n_stft=100, wndw='hamming'):
    dt = time[1] - time[0]

    # 入力された n_wndw を t_wndwで設定した幅より小さい，かつ，2の累乗個に設定する
    n_wndw = int(2 ** (np.floor(np.log2(t_wndw / dt))))
    t_wndw = n_wndw * dt  # recalculate t_wndw
    n_freq = n_wndw

    # 周波数
    freq_sp = np.fft.fftfreq(n_wndw, dt)

    # スペクトルを計算する時刻を決める
    m = len(time) - n_wndw
    indxs = np.zeros(n_stft, dtype=int)
    for i in range(n_stft):
        indxs[i] = int(m / (n_stft + 1) * (i + 1)) + n_wndw // 2

    tm_sp = time[indxs]  # DFTをかける時刻の配列

    # スペクトログラムを計算する
    # スペクトルは indxs[i] - n_wndw //2 + 1 ~ indxs[i] + n_wndw//2 の n_wndw 幅で行う
    sp = np.zeros((n_freq, n_stft), dtype=complex)  # スペクトログラムの2次元ndarray

    if wndw == 'hamming':
        wndw = np.hamming(n_wndw)  # hamming窓
    if wndw == 'hanning':
        wndw = np.hanning(n_wndw)  # hanning窓
    if wndw == 'rect':
        wndw = np.ones(n_wndw)  # 矩形窓

    for i in range(n_stft):
        indx = indxs[i] - n_wndw // 2 + 1
        sp[:, i] = np.fft.fft(wndw * signal[indx:indx + n_wndw], n_wndw) / np.sqrt(n_wndw)

    return tm_sp, freq_sp[freq_sp >= 0], sp[freq_sp >= 0]


def cross_spectrogram(sp1, sp2):
    if sp1.ndim == 2:
        xsp = calc_cross_spectrogram(sp1, sp2)
    if sp1.ndim == 3:
        xsp = np.zeros_like(sp1, dtype=complex)
        for i in range(sp1.shape[0]):
            xsp[i, :] = calc_cross_spectrogram(sp1[i, :], sp2[i, :])
    if sp1.ndim == 4:
        xsp = np.zeros_like(sp1, dtype=complex)
        for i in range(sp1.shape[0]):
            for j in range(sp1.shape[1]):
                xsp[i, j, :] = calc_cross_spectrogram(sp1[i, j, :], sp2[i, j, :])
    if sp1.ndim == 5:
        xsp = np.zeros_like(sp1, dtype=complex)
        for i in range(sp1.shape[0]):
            for j in range(sp1.shape[1]):
                for k in range(sp1.shape[2]):
                    xsp[i, j, k, :] = calc_cross_spectrogram(sp1[i, j, k, :], sp2[i, j, k, :])
    return xsp


def calc_cross_spectrogram(sp1, sp2):
    xsp = sp1 * np.conjugate(sp2)

    return xsp


def smoothing(sp):
    if sp.ndim == 2:
        sp_smthd = calc_smoothing(sp)
    if sp.ndim == 3:
        sp_smthd = np.zeros_like(sp)
        for i in range(sp.shape[0]):
            sp_smthd[i, :] = calc_smoothing(sp[i, :])
    if sp.ndim == 4:
        sp_smthd = np.zeros_like(sp)
        for i in range(sp.shape[0]):
            for j in range(sp.shape[1]):
                sp_smthd[i, j, :] = calc_smoothing(sp[i, j, :])
    if sp.ndim == 5:
        sp_smthd = np.zeros_like(sp)
        for i in range(sp.shape[0]):
            for j in range(sp.shape[1]):
                for k in range(sp.shape[2]):
                    sp_smthd[i, j, k, :] = calc_smoothing(sp[i, j, k, :])
    return sp_smthd


# 時間または周波数方向に三角窓で平滑化する関数
def calc_smoothing(sp):
    n_f, n_t = sp.shape
    sp_smthd = np.zeros_like(sp)

    for i in range(n_t):
        # krnl = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        krnl = np.array([1.0, 2.0, 1.0])
        sp_smthd[:, i] = np.convolve(sp[:, i], krnl, mode='same') / np.sum(krnl)

    for j in range(n_f):
        # krnl = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        krnl = np.array([1.0, 2.0, 1.0])
        sp_smthd[j, :] = np.convolve(sp_smthd[j, :], krnl, mode='same') / np.sum(krnl)

    return sp_smthd


def coherence(xsp, sp1, sp2):
    if sp1.ndim == 2:
        coh, phs = calc_coherence(xsp, sp1, sp2)
    if sp1.ndim == 3:
        coh = np.zeros_like(sp1)
        phs = np.zeros_like(sp1)
        print(coh.shape)
        for i in range(sp1.shape[0]):
            coh[i, :], phs[i, :] = calc_coherence(xsp[i, :], sp1[i, :], sp2[i, :])
    if sp1.ndim == 4:
        coh = np.zeros_like(sp1)
        phs = np.zeros_like(sp1)
        for i in range(sp1.shape[0]):
            for j in range(sp1.shape[1]):
                coh[i, j, :], phs[i, j, :] = calc_coherence(xsp[i, j, :], sp1[i, j, :], sp2[i, j, :])
    if sp1.ndim == 5:
        coh = np.zeros_like(sp1)
        phs = np.zeros_like(sp1)
        for i in range(sp1.shape[0]):
            for j in range(sp1.shape[1]):
                for k in range(sp1.shape[2]):
                    coh[i, j, k, :], phs[i, j, k, :] = calc_coherence(xsp[i, j, k, :], sp1[i, j, k, :], sp2[i, j, k, :])
    return coh, phs


def calc_coherence(xsp, sp1, sp2):
    sp1_pw_smthd = smoothing(np.abs(sp1) ** 2)  # 平滑化
    sp2_pw_smthd = smoothing(np.abs(sp2) ** 2)  # 平滑化
    xsp_smthd = calc_smoothing(xsp)  # 平滑化
    coh = np.abs(xsp_smthd) ** 2 / (sp1_pw_smthd * sp2_pw_smthd)  # （二乗）コヒーレンス
    phs = np.rad2deg(np.arctan2(np.imag(xsp_smthd), np.real(xsp_smthd)))  # フェイズ
    return coh, phs


if __name__ == "__main__":
    # # ボイスメモで収録したm4aファイルを読み込む
    # sounds = AudioSegment.from_file('dataset/audio/aiueo.m4a', 'm4a')
    # print(f'channel: {sounds.channels}')
    # print(f'frame rate: {sounds.frame_rate}')
    # print(f'duration: {sounds.duration_seconds} s')
    #
    # # チャンネルが2（ステレオ) の場合，L/R交互にデータが入っているので，二つおきに読み出す。
    # sig = np.array(sounds.get_array_of_samples())[::sounds.channels]
    # dt = 1.0 / sounds.frame_rate  # サンプリング時間
    # tms = 0.0
    # tme = sounds.duration_seconds  # サンプル終了時刻
    #
    # tm = np.linspace(tms, tme, len(sig), endpoint=False)  # 時間ndarrayを作成
    #
    # # ウィンドウ幅，STFTを施す数の設定
    # t_wndw = 100.0e-3  # 100 mili-second
    # n_stft = 100  # number of STFT
    # freq_upper = 2.0e3  # 表示する周波数の上限
    #
    # # STFTを計算
    # tm_sp, freq_sp, sp = stft(sig, tm, t_wndw=t_wndw, n_stft=n_stft)
    # # 平滑化
    # sp_smthd = calc_smoothing(sp)
    #
    # # 解析結果の可視化
    # figsize = (210 / 25.4, 294 / 25.4)
    # dpi = 200
    # fig = plt.figure(figsize=figsize, dpi=dpi)
    #
    # # --- 図の設定 (全体)
    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['xtick.top'] = True
    # plt.rcParams['xtick.major.size'] = 6
    # plt.rcParams['xtick.minor.size'] = 3
    # plt.rcParams['xtick.minor.visible'] = True
    # plt.rcParams['ytick.direction'] = 'in'
    # plt.rcParams['ytick.right'] = True
    # plt.rcParams['ytick.major.size'] = 6
    # plt.rcParams['ytick.minor.size'] = 3
    # plt.rcParams['ytick.minor.visible'] = True
    # plt.rcParams["font.size"] = 14
    # plt.rcParams['font.family'] = 'Helvetica'
    #
    # # 窓関数幅をプロット上部に記載
    # fig.text(0.10, 0.95, f't_wndw = {t_wndw} s, ')
    #
    # # プロット枠 (axes) の設定
    # ax1 = fig.add_axes([0.15, 0.55, 0.70, 0.3])
    # ax_sp1 = fig.add_axes([0.15, 0.2, 0.70, 0.30])
    # cb_sp1 = fig.add_axes([0.87, 0.2, 0.02, 0.30])
    #
    # # 元データのプロット
    # ax1.set_xlim(tms, tme)
    # ax1.set_xlabel('')
    # ax1.tick_params(labelbottom=False)
    # ax1.set_ylabel('x')
    #
    # ax1.plot(tm, sig, c='black')
    #
    # # スペクトログラムのプロット
    # ax_sp1.set_xlim(tms, tme)
    # ax_sp1.set_xlabel('time (s)')
    # ax_sp1.tick_params(labelbottom=True)
    # ax_sp1.set_ylim(0, freq_upper)
    # ax_sp1.set_ylabel('frequency\n(Hz)')
    #
    # norm = mpl.colors.Normalize(vmin=np.log10(np.abs(sp[freq_sp < freq_upper, :]) ** 2).min(),
    #                             vmax=np.log10(np.abs(sp[freq_sp < freq_upper, :]) ** 2).max())
    # cmap = mpl.cm.jet
    #
    # ax_sp1.contourf(tm_sp, freq_sp, np.log10(np.abs(sp) ** 2),
    #                 norm=norm,
    #                 levels=256,
    #                 cmap=cmap)
    #
    # ax_sp1.text(0.99, 0.97, "spectrogram", color='white', ha='right', va='top',
    #             path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
    #                           path_effects.Normal()],
    #             transform=ax_sp1.transAxes)
    #
    # mpl.colorbar.ColorbarBase(cb_sp1, cmap=cmap,
    #                           norm=norm,
    #                           orientation="vertical",
    #                           label='$\log_{10}|X/N|^2$')
    #
    # plt.show()

    # テスト信号の生成
    dt = 1.0e-4
    tms = 0.0
    tme = 2.0
    tm01 = np.arange(tms, tme, dt)
    tm02 = tm01

    # np.random.seed(1234)
    sig01 = np.random.randn(len(tm02)) * 0.05
    sig01[tm01 < 0.3] += np.sin(2.0 * np.pi * 5.0e1 * tm01[tm01 < 0.3])
    sig01[(0.5 < tm01) & (tm01 < 0.8)] += np.sin(2.0 * np.pi * 1.0e2 * tm01[(0.5 < tm01) & (tm01 < 0.8)])
    sig01[(1.0 < tm01) & (tm01 < 1.3)] += np.sin(2.0 * np.pi * 1.5e2 * tm01[(1.0 < tm01) & (tm01 < 1.3)])
    sig01[(1.5 < tm01) & (tm01 < 1.8)] += np.sin(2.0 * np.pi * 2.0e2 * tm01[(1.5 < tm01) & (tm01 < 1.8)])

    sig02 = np.sin(2.0 * np.pi * 5.0e1 * tm02) + 0.1 * np.sin(
        2.0 * np.pi * 1.5e2 * tm02 + np.pi / 2.0) + np.random.randn(len(tm02)) * 0.05

    # ウィンドウ幅，STFTを施す数の設定
    t_wndw = 100.0e-3  # 100 milisecond
    n_stft = 100  # number of STFT
    freq_upper = 3.0e2  # 表示する周波数の上限

    # STFTを実行
    tm_sp01, freq_sp01, sp01 = stft(sig01, tm01, t_wndw=t_wndw, n_stft=n_stft, wndw='hanning')
    tm_sp02, freq_sp02, sp02 = stft(sig02, tm02, t_wndw=t_wndw, n_stft=n_stft, wndw='hanning')
    # クロススペクトル
    xsp = cross_spectrogram(sp01, sp02)
    # コヒーレンスとフェイズ
    coh, phs = coherence(xsp, sp01, sp02)

    # 結果のプロット
    # 解析結果の可視化
    figsize = (210 / 25.4, 294 / 25.4)
    dpi = 200
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # 図の設定 (全体)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.major.size'] = 6
    plt.rcParams['xtick.minor.size'] = 3
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.major.size'] = 6
    plt.rcParams['ytick.minor.size'] = 3
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams["font.size"] = 14
    plt.rcParams['font.family'] = 'Helvetica'

    # 窓関数幅をプロット上部に記載
    fig.text(0.10, 0.95, f't_wndw = {t_wndw} s')

    # プロット枠の設定
    ax01 = fig.add_axes([0.125, 0.79, 0.70, 0.08])
    ax02 = fig.add_axes([0.125, 0.59, 0.70, 0.08])

    ax_sp01 = fig.add_axes([0.125, 0.68, 0.70, 0.10])
    cb_sp01 = fig.add_axes([0.85, 0.68, 0.02, 0.10])
    ax_sp02 = fig.add_axes([0.125, 0.48, 0.70, 0.10])
    cb_sp02 = fig.add_axes([0.85, 0.48, 0.02, 0.10])

    ax_xsp = fig.add_axes([0.125, 0.33, 0.70, 0.10])
    cb_xsp = fig.add_axes([0.85, 0.33, 0.02, 0.10])
    ax_coh = fig.add_axes([0.125, 0.22, 0.70, 0.10])
    cb_coh = fig.add_axes([0.85, 0.22, 0.02, 0.10])
    ax_phs = fig.add_axes([0.125, 0.10, 0.70, 0.10])
    cb_phs = fig.add_axes([0.85, 0.10, 0.02, 0.10])

    # ---------------------------
    # テスト信号 sig01 のプロット
    ax01.set_xlim(tms, tme)
    ax01.set_xlabel('')
    ax01.tick_params(labelbottom=False)
    ax01.set_ylabel('x (sig01)')

    ax01.plot(tm01, sig01, c='black')

    # ---------------------------
    # テスト信号 sig02 のプロット
    ax02.set_xlim(tms, tme)
    ax02.set_xlabel('')
    ax02.tick_params(labelbottom=False)
    ax02.set_ylabel('y (sig02)')

    ax02.plot(tm02, sig02, c='black')

    # ---------------------------
    # テスト信号 sig01 のスペクトログラムのプロット
    ax_sp01.set_xlim(tms, tme)
    ax_sp01.set_xlabel('')
    ax_sp01.tick_params(labelbottom=False)
    ax_sp01.set_ylim(0, freq_upper)
    ax_sp01.set_ylabel('frequency\n(Hz)')

    norm = mpl.colors.Normalize(vmin=np.log10(np.abs(sp01[freq_sp01 < freq_upper, :]) ** 2).min(),
                                vmax=np.log10(np.abs(sp01[freq_sp01 < freq_upper, :]) ** 2).max())
    cmap = mpl.cm.jet

    ax_sp01.contourf(tm_sp01, freq_sp01, np.log10(np.abs(sp01) ** 2),
                     norm=norm, levels=256, cmap=cmap)

    ax_sp01.text(0.99, 0.97, "spectrogram", color='white', ha='right', va='top',
                 path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                               path_effects.Normal()],
                 transform=ax_sp01.transAxes)

    mpl.colorbar.ColorbarBase(cb_sp01, cmap=cmap, norm=norm,
                              orientation="vertical",
                              label='$\log_{10}|X/N|^2$')

    # ---------------------------
    # テスト信号 sig02 のスペクトログラムのプロット
    ax_sp02.set_xlim(tms, tme)
    ax_sp02.set_xlabel('')
    ax_sp02.tick_params(labelbottom=True)
    ax_sp02.set_ylim(0, freq_upper)
    ax_sp02.set_ylabel('frequency\n(Hz)')

    norm = mpl.colors.Normalize(vmin=np.log10(np.abs(sp02[freq_sp02 < freq_upper, :]) ** 2).min(),
                                vmax=np.log10(np.abs(sp02[freq_sp02 < freq_upper, :]) ** 2).max())
    ax_sp02.contourf(tm_sp02, freq_sp02, np.log10(np.abs(sp02) ** 2),
                     norm=norm, levels=256, cmap=cmap)

    ax_sp02.text(0.99, 0.97, "spectrogram", color='white', ha='right', va='top',
                 path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                               path_effects.Normal()],
                 transform=ax_sp02.transAxes)

    mpl.colorbar.ColorbarBase(cb_sp02, cmap=cmap, norm=norm,
                              orientation="vertical",
                              label='$\log_{10}|Y/N|^2$')

    # ---------------------------
    # テスト信号 sig01 と sig02 のクロススペクトルのプロット
    ax_xsp.set_xlim(tms, tme)
    ax_xsp.set_xlabel('')
    ax_xsp.tick_params(labelbottom=False)
    ax_xsp.set_ylim(0, freq_upper)
    ax_xsp.set_ylabel('frequency\n(Hz)')

    norm = mpl.colors.Normalize(vmin=np.log10(np.abs(xsp[freq_sp02 < freq_upper, :])).min(),
                                vmax=np.log10(np.abs(xsp[freq_sp02 < freq_upper, :])).max())

    ax_xsp.contourf(tm_sp01, freq_sp01, np.log10(np.abs(xsp)),
                    norm=norm, levels=256, cmap=cmap)

    ax_xsp.text(0.99, 0.97, "cross-spectrum", color='white', ha='right', va='top',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                              path_effects.Normal()],
                transform=ax_xsp.transAxes)

    mpl.colorbar.ColorbarBase(cb_xsp, cmap=cmap, norm=norm,
                              orientation="vertical",
                              label='$\log_{10}|XY^*/N^2|$')

    # ---------------------------
    # テスト信号 sig01 と sig02 のコヒーレンスのプロット
    ax_coh.set_xlim(tms, tme)
    ax_coh.set_xlabel('')
    ax_coh.tick_params(labelbottom=False)
    ax_coh.set_ylim(0, freq_upper)
    ax_coh.set_ylabel('frequency\n(Hz)')

    norm = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
    ax_coh.contourf(tm_sp01, freq_sp01, coh,
                    norm=norm, levels=10, cmap=cmap)

    ax_coh.text(0.99, 0.97, "coherence", color='white', ha='right', va='top',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                              path_effects.Normal()],
                transform=ax_coh.transAxes)

    mpl.colorbar.ColorbarBase(cb_coh, cmap=cmap, norm=norm,
                              boundaries=np.linspace(0, 1, 11),
                              orientation="vertical",
                              label='coherence')

    # ---------------------------
    # テスト信号 sig01 と sig02 のフェイズのプロット
    ax_phs.set_xlim(tms, tme)
    ax_phs.set_xlabel('time (s)')
    ax_phs.tick_params(labelbottom=True)
    ax_phs.set_ylim(0, freq_upper)
    ax_phs.set_ylabel('frequency\n(Hz)')

    norm = mpl.colors.Normalize(vmin=-180.0, vmax=180.0)
    cmap = mpl.cm.hsv
    ax_phs.contourf(tm_sp01, freq_sp01, np.where(coh >= 0.75, phs, None),
                    norm=norm, levels=16, cmap=cmap)

    ax_phs.text(0.99, 0.97, "phase", color='white', ha='right', va='top',
                path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                              path_effects.Normal()],
                transform=ax_phs.transAxes)

    mpl.colorbar.ColorbarBase(cb_phs, cmap=cmap,
                              norm=norm,
                              boundaries=np.linspace(-180.0, 180.0, 17),
                              orientation="vertical",
                              label='phase (deg)')
    plt.show()