[A,delimiterOut] = importdata('out.csv')
% disp(A.data(:,2));

y = A.data(:,4);
u = A.data(:,3);

Ts = 0.005
data = iddata(y,u,Ts) % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示
% plot(data)

% 相関解析法によるインパルス応答の推定
% ir = impulse(data)
% sr = step(data)
% plot(sr)

% GS = spa(data(1:100));
% h = bodeplot(GS); % bodeplot returns a plot handle, which bode does not
% ax = axis; axis([0.1 10 ax(3:4)])

% nx = 4;
% sys = ssest(data,nx);

% compare(data, sys)

m1 = pem(data)
m2 = tf(m1)
m3 = d2c(m2)

figure();
compare(data, m1)