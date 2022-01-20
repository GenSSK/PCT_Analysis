p_filename = "2021-12-09_doutei_dec_p.csv"  %pitch同定用データ
r_filename = "2021-12-09_doutei_dec_r.csv"  %roll同定用データ
t_p_filename = "2021-12-09_test_dec_p.csv"  %pitchテスト用データ
t_r_filename = "2021-12-09_test_dec_r.csv"  %rollテスト用データ

dir = "J:\マイドライブ\program\ARCS-PCT\data\doutei_PCT\\"    %ファイルの置き場所

p_dir = fullfile(dir, p_filename);       %pitch同定用データのディレクトリ取得
t_p_dir = fullfile(dir, t_p_filename);   %roll同定用データのディレクトリ取得
r_dir = fullfile(dir, r_filename);       %pitchテスト用データのディレクトリ取得
t_r_dir = fullfile(dir, t_r_filename);   %rollテスト用データのディレクトリ取得

%pitchとrollでどちらかを選択（コメントアウトする）
%pitch
%[A,delimiterOut] = importdata(p_dir);
%[B,delimiterOut] = importdata(t_p_dir);
%roll
[A,delimiterOut] = importdata(r_dir);
[B,delimiterOut] = importdata(t_r_dir);

Ts = A.data(2, 1) - A.data(1, 1)    %サンプリング時間（同定用信号の更新時間）

%デトレンドして間引いて代入，データ作成
thm = detrend(A.data(:, 5));
wm = detrend(A.data(:, 4));
am = detrend(A.data(:, 3));
tad = detrend(A.data(:, 2));

t_thm = detrend(B.data(:, 5));
t_wm = detrend(B.data(:, 4));
t_am = detrend(B.data(:, 3));
t_tad = detrend(B.data(:, 2));


data = iddata(wm, tad, Ts); % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示
test = iddata(t_wm, t_tad, Ts); % iddata オブジェクトの生成% y:出力，u:入力，Ts:サンプリング周期 % 入出力データの表示

% テストデータ確認
figure();
plot(data);
%figure();
%plot(test);


%identification
m = ssest(data, 1);
m_d = ssest(data, 1, 'DisturbanceModel','none');
mtf = tfest(data, 2, 2); % transfer function with 2 zeros and 2 poles
m_tf = tf(m)


mx = arx(data,[1 1 1]);
m2 = tf(mx);
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