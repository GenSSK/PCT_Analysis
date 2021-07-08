[A,delimiterOut] = importdata('motor_fanalysis.csv')

p_thm = A.data(1:50000, 5);
p_wm = A.data(1:50000, 4);
p_am = A.data(1:50000, 3);
p_iq = A.data(1:50000, 2);

r_thm = A.data(1:50000, 9);
r_wm = A.data(1:50000, 8);
r_am = A.data(1:50000, 7);
r_iq = A.data(1:50000, 6);

Ts = 0.0001
data = iddata(p_wm, p_iq, Ts) % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示

data = detrend(data)

m = ssest(data);

h = bodeplot(m)