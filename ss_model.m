[A,delimiterOut] = importdata('data.csv')
[B,delimiterOut] = importdata('test.csv')
% disp(A.data(:,2));

y1 = A.data(:,5);
u1 = A.data(:,3);

y2 = A.data(:,5);
u2 = A.data(:,3);

Ts = 0.005
data = iddata(y1,u2,Ts) % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示
test = iddata(y2,u2,Ts) % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示


% 状態空間モデルの同定
% u = U4_pre;
% y = [Y4_ang_pre Y4_vel_pre];
% z = iddata(y,u,0.005);
% u_post = U4_post;
% y_post = [Y4_ang_post Y4_vel_post];
% z_post = iddata(y_post,u_post,0.005);

As = [0 1;0 NaN];
Bs = [0;NaN];
Cs = [0 0;0 1];
Ds = [0; 0];
Ks = [0 0;0 0];
X0s =[0 ;0];

A = [0 1; 0 -1];
B = [0 ; 0.28];
C = [0 ; 1];
D = zeros(2,1);

ms = idss(A,B,C,D);

setstruc(ms , As , Bs , Cs, Ds, Ks, X0s);
set(ms,'Ts',0);

dcmodel = pem(data, ms ,'trace', 'on')
figure();
% compare(z,dcmodel);
compare(test,dcmodel);