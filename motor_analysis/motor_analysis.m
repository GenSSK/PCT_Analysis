%[A,delimiterOut] = importdata('motor_analysis.csv')
[A,delimiterOut] = importdata('motor_analysis_StepWave.csv')
%[A,delimiterOut] = importdata('motor_analysis_StepWave_DOB_gd100.csv')
%[A,delimiterOut] = importdata('motor_analysis_StepWave_DOB_gd300.csv')

p_thm = A.data(1:40000, 5);
p_wm = A.data(1:40000, 4);
p_am = A.data(1:40000, 3);
p_iq = A.data(1:40000, 2);

r_thm = A.data(1:40000, 9);
r_wm = A.data(1:40000, 8);
r_am = A.data(1:40000, 7);
r_iq = A.data(1:40000, 6);

Ts = 0.0001

data = iddata(r_thm, r_iq, Ts) % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示

figure();
L = 50000;             % Length of signal
t = (0:L-1)*Ts;        % Time vector
Y = fft(r_am);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = 1/Ts*(0:(L/2))/L;
plot(f,P1)
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

figure();
plot(data)

figure();
%GS = spa(data)
GS = arx(data, [10 1 1])
h = bodeplot(GS); % bodeplot returns a plot handle, which bode does not
%ax = axis; axis([0.1 10 ax(3:4)])

%A = [0 1; 0 0];
%B = [0 ; 0.076262298214];
%C = [1 0];
%D = [0];
%
%ms = idss(A, B, C, D);

num = [1];
den = [0.076262298214 0 0];
G = tf(num, den, Ts)

figure();
compare(data, GS, G)