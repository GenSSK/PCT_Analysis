[A,delimiterOut] = importdata('balanced_data.csv')
[B,delimiterOut] = importdata('balanced_test.csv')

p_thm = detrend(A.data(:, 5));
p_iq = detrend(A.data(:, 2));

r_thm = detrend(A.data(:, 9));
r_iq = detrend(A.data(:, 6));

t_p_thm = detrend(B.data(1:512, 5));
t_p_iq = detrend(B.data(1:512, 2));

t_r_thm = detrend(B.data(1:512, 9));
t_r_iq = detrend(B.data(1:512, 6));

Ts = 0.01

data = iddata(p_thm, p_iq, Ts) % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示
test = iddata(t_p_thm, t_p_iq, Ts) % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示

%figure();
%plot(data);
%figure();
%plot(test);

%identification
m = ssest(data, 1)
%m = ssest(data, 1, 'DisturbanceModel','none')
%mtf = tfest(data, 2, 2) % transfer function with 2 zeros and 2 poles


%mx = arx(data,[1 1 1])
%m2 = tf(mx)
%m3 = d2c(m2)
%b = [-0.04532];
%a = [1 -0.01282];
%[A,B,C,D] = tf2ss(b,a)

figure();
%h = bodeplot(m3)
%compare(test,m,mtf,mx,1)
compare(test, m, 1)


% only kt and inertia
%As = [0 1; 0 0];
%Bs = [0; NaN];
%Cs = [1 0];
%Ds = [0];
%Ks = [0; 0];
%X0s =[0; 0];
%
%A = [0 1; 0 0];
%B = [0 ; 1];
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
%compare(test, SPMSM, 1);