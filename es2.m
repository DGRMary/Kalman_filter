clear all 
close all 
clc

load data2a.mat

rng('default');

N = length(u);

A = [0 0 -10;
    0.02 0 0;
    0.2485 -3.7885 2.7258];
B = [0 0 0.026]';
C = [0.18 0.5 2];
D = 0;

[n nn] = size(A);

Bv1 = 0.0004*[10 1 1]';
V1 = Bv1*Bv1';
V2 = 0.0004;

x(:,1) = [-500 -10 50]';
N0_vector = [0 10 100];
P = eye(n);

%system
for t = 1:N,
    v1(:,t) = mvnrnd(zeros(n,1),V1); 
    v2(t) = sqrt(V2)*randn;
    x(:,t+1) = A*x(:,t)+B*u(t)+v1(:,t);
    y(t) = C*x(:,t)+v2(t);
end

%% steady state Kinf in predictor form
sys = ss(A,[B,eye(n)],C,[D,zeros(1,n)],1);
[kalman_predictor,Kbar,Pbar,K0bar] = kalman(sys,V1,V2,0);
K0bar,Kbar

xh_ss_pc(:,1) = zeros(n,1);
for t = 1:N,
    P0 = (eye(n)-K0bar*C)*P;
    yh_ss_pc(t) = C*xh_ss_pc(:,t);
    e(t) = y(t)-yh_ss_pc(t);
    xf_ss_pc(:,t) = xh_ss_pc(:,t)+K0bar*e(t);
    yf_ss_pc(t) = C*xf_ss_pc(:,t);
    P = A*P0*A'+V1;
    xh_ss_pc(:,t+1) = A*xf_ss_pc(:,t)+B*u(t);
end

%% kalman filter Fd in standard form
xh(:,1) = zeros(n,1);
P = eye(n);

for t = 1:N,
    yh(t) = C*xh(:,t);
    ed(t) = y(t)-yh(t);
    K0 = P*C'*inv(C*P*C'+V2);
    K = A*K0;
    P = A*P*A'+V1-K*(C*P*C'+V2)*K';
    xh(:,t+1) = A*xh(:,t)+B*u(t)+K*ed(t);
    xf(:,t) = xh(:,t)+K0*ed(t);
    yf(t)= C*xf(:,t);
end
K, K0

%% RMSE

for ind = 1:length(N0_vector),
    N0 = N0_vector(ind);
    for k = 1:3,
        RMSE_xh_ss_pc(k,ind) = norm(x(k,N0+1:N)-xh_ss_pc(k,N0+1:N))/sqrt(N-N0);
        RMSE_xf(k,ind) = norm(x(k,N0+1:N)-xf(k,N0+1:N))/sqrt(N-N0);
    end
    RMSE_yh_ss_pc(ind) = norm(y(N0+1:N)-yh_ss_pc(N0+1:N))/sqrt(N-N0);
    RMSE_yf(ind) = norm(y(N0+1:N)-yf(N0+1:N))/sqrt(N-N0);
end

fprintf ('\n N0 = %d, N0 = %d, N0 = %d \n',N0_vector(1),N0_vector(2),N0_vector(3))
RMSE_xh_ss_pc,RMSE_xf
fprintf ('\n N0 = %d, N0 = %d, N0 = %d \n',N0_vector(1),N0_vector(2),N0_vector(3))
RMSE_yh_ss_pc,RMSE_yf

T = 1:N;
for k = 1:n,
    figure,plot(T,x(k,T),'b',T,xh_ss_pc(k,T),'g',T,xf(k,T),'r')
end
figure,plot(T,y(T),'b',T,yh_ss_pc(T),'g',T,yf(T),'r')