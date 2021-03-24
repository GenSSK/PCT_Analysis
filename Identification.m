[A,delimiterOut] = importdata('data.csv')
[B,delimiterOut] = importdata('test.csv')

% disp(A.data(:,2));

y1 = A.data(:,5);
u1 = A.data(:,3);

y2 = detrend(B.data(1:240, 5));
u2 = detrend(B.data(1:240, 3));

Ts = 0.005
data = iddata(y1,u2,Ts) % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示
test = iddata(y2,u2,Ts) % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示

figure(1)
plot(test)

% 相関解析法によるインパルス応答の推定
% ir = impulse(data, 'sd', 3)
% % sr = step(data)
% figure(1);
% plot(ir)

m = ssest(data)
% GS = spa(data);
% h = bodeplot(GS); % bodeplot returns a plot handle, which bode does not
% ax = axis; axis([0.1 10 ax(3:4)])`

% nx = 4;
% sys = ssest(data,nx);
% compare(data, sys)

% m1 = pem(data)
% m2 = tf(m1)
% m3 = d2c(m2)

% figure();
% compare(data, m1)

figure();
mtf = tfest(data, 2, 2) % transfer function with 2 zeros and 2 poles
mx = arx(data,[1 1 1])
m2 = tf(mx)
m3 = d2c(m2)
h = bodeplot(m3)
figure();
compare(test,m,mtf,mx,1)

% md1 = tfest(data,2,2,'Ts',Ts)  % two poles and 2 zeros (counted as roots of polynomials in z^-1)
% md2 = oe(data,[2 2 1]) 
% compare(test, md1, md2)

% resid(test,md2) % plots the result of residual analysis

% am2 = armax(data,[2 2 2 1])  % 2nd order ARMAX model
% bj2 = bj(data,[2 2 2 2 1])   % 2nd order BOX-JENKINS model

% clf
% figure(1);
% compare(test,am2,md2,bj2,mx,mtf,m)

% figure(2);
% compare(test,am2,md2,bj2,mx,mtf,m,1)

% figure(3);
% resid(test,am2)