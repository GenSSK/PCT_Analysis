B = [0 1 0.5];
A = [1 -1.5 0.7];
m0 = idpoly(A,B,[1 -1 0.2],'Ts',0.25,'Variable','q^-1'); % The sample time is 0.25 s.

% モデルのシミュレーション
prevRng = rng(12,'v5normal');
u = idinput(350,'rbs');  %Generates a random binary signal of length 350
u = iddata([],u,0.25);   %Creates an IDDATA object. The sample time is 0.25 sec.
y = sim(m0,u,'noise')    %Simulates the model's response for this

rng(prevRng);

z = [y,u];

h = gcf; set(h,'DefaultLegendLocation','best')
h.Position = [100 100 780 520];
plot(z(1:100));

ze = z(1:200);
zv = z(201:350);

% スペクトル解析の実行
GS = spa(ze);
clf
h = bodeplot(GS); % bodeplot returns a plot handle, which bode does not
ax = axis; axis([0.1 10 ax(3:4)])

% パラメトリック状態空間モデルの推定
m = ssest(ze) % The order of the model will be chosen automatically
bodeplot(m,GS)
ax = axis; axis([0.1 10 ax(3:4)])

% 簡単な伝達関数の推定
mtf = tfest(ze, 2, 2) % transfer function with 2 zeros and 2 poles
mx = arx(ze,[2 2 1])
compare(zv,m,mtf,mx)