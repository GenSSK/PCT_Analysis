[A,delimiterOut] = importdata('data.csv');
[B,delimiterOut] = importdata('test.csv');
% disp(A.data(:,2));

y1 = A.data(:, 5);
u1 = A.data(:, 3);

y2 = B.data(1:240, 5);
u2 = B.data(1:240, 3);

Ts = 0.005;
data = iddata(y1,u2,Ts); % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示
test = iddata(y2,u2,Ts); % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示

% only kt and inertia
% As = [0 1; 0 0];
% Bs = [0; NaN];
% Cs = [1 0];
% Ds = [0];
% Ks = [0; 0];
% X0s =[0; 0];
% 
% A = [0 1; 0 0];
% B = [0 ; 1];
% C = [1 0];
% D = [0];

% using D.E. of electronic
As = [0 0 0; 0 0 1; NaN 0 NaN];
Bs = [0; 0; NaN];
Cs = [0 1 0];
Ds = [0];
Ks = [0; 0; 0];
X0s =[0; 0; 0];

A = [0 0 0; 0 0 1; 1 0 1];
B = [0 ; 0; 1];
C = [0 1 0];
D = [0];

ms = idss(A,B,C,D);

setstruc(ms, As, Bs, Cs, Ds, Ks, X0s);
set(ms,'Ts',0);

% opt = ssestOptions('EnforceStability', true)

opt = ssestOptions;
opt.EnforceStability = true;
opt.OutputWeight = trace;
opt.Display = on;
SPMSM = pem(data, ms, opt);
% SPMSM = pem(data, ms, 'trace', 'on', 'EnforceStability = TRUE');
figure();
% compare(z,dcmodel);
compare(test, SPMSM, 1);