[A,delimiterOut] = importdata('pitch.csv')
[B,delimiterOut] = importdata('pitch_test.csv')
%[A,delimiterOut] = importdata('roll.csv')
%[B,delimiterOut] = importdata('roll_test.csv')

Ts = 0.01
dec = Ts / 0.0001

thm = decimate(detrend(A.data(10000:50000, 5)), dec);
wm = decimate(detrend(A.data(10000:50000, 4)), dec);
am = decimate(detrend(A.data(10000:50000, 3)), dec);
tad = decimate(detrend(A.data(10000:50000, 2)), dec);

t_thm = decimate(detrend(B.data(10000:50000, 5)), dec);
t_wm = decimate(detrend(B.data(10000:50000, 4)), dec);
t_am = decimate(detrend(B.data(10000:50000, 3)), dec);
t_tad = decimate(detrend(B.data(10000:50000, 2)), dec);

data = iddata(wm, tad, Ts) % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示
test = iddata(t_wm, t_tad, Ts) % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示

% テストデータ確認
%figure();
%plot(data);
%figure();
%plot(test);


%identification
m = ssest(data, 1)
m_d = ssest(data, 1, 'DisturbanceModel','none')
mtf = tfest(data, 2, 2) % transfer function with 2 zeros and 2 poles


mx = arx(data,[1 1 1])
m2 = tf(mx)
m3 = d2c(m2)
%b = [-0.04532];
%a = [1 -0.01282];
%[A,B,C,D] = tf2ss(b,a)

%tfx = tfestimate(p_thm, p_iq)
sysTF = tfest(data,1,0,nan)
%figure();
%h = bodeplot(m3)
figure();
compare(test,m,mtf,mx,m_d,sysTF, 1)

m_tf = tf(m)



% only kt and inertia
%As = [0 1; 0 0];
%Bs = [0; NaN];
%Cs = [1 0];
%Ds = [0];
%Ks = [0; 0];
%X0s =[0; 0];
%
%A = [0 1; 0 0];
%B = [0 ; 0.28];
%C = [1 0];
%D = [0];
%
%ms = idss(A, B, C, D);
%
%setstruc(ms, As, Bs, Cs, Ds, Ks, X0s)
%set(ms,'Ts', 0)

% opt = ssestOptions('EnforceStability', true)

%opt = ssestOptions;
%opt.EnforceStability = true;
%opt.OutputWeight = trace;
%opt.Display = on;
%SPMSM = pem(data, ms)
%SPMSM = pem(data, ms, 'trace', 'on')


%figure();
%compare(data, SPMSM, 1);