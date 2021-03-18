syms Ke L J R Kt s
A = [-(Ke * Kt)/(J * R) 0 -(Kt)/(J * R * L); 1 0 0; 0 0 0];
I = eye(3);

W = inv(s * I - A)

B = [(Kt)/(J * R); 0; 0];
C = [0 1 0];

G = C * W * B