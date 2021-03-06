%% Cleaning
clear('all')
clc
close('all')

%% Inputs
rng(1585); % For replication
N = 1e5; % Number of simulations
U = rand(N,2); % Indep. U(0,1) random matrix of order (N,2) 

%% Section a)
% Clayton
theta1 = 2;  % Small dependence relationship
theta2 = 10; % Greater dependence relationship

w1_clay = clayton(theta1,U);
w2_clay = clayton(theta2,U);

% Gaussian
rho1 = -0.7;
corr1 = [1, rho1;
         rho1, 1];
rho2 = 0.6;
corr2 = [1, rho2;
         rho2, 1];
w1_gauss = simuN(corr1,U);
w2_gauss = simuN(corr2,U);

% Plotting

figure(1);
subplot(2,1,1)
plot(w1_clay(:,1), w1_clay(:,2), 'o','MarkerSize',0.5)
title('Theta = 2')

subplot(2,1,2)
plot(w2_clay(:,1), w2_clay(:,2), 'o','MarkerSize',0.5)
title('Theta = 10')

figure(2);
subplot(2,1,1)
plot(w1_gauss(:,1), w1_gauss(:,2), 'o','MarkerSize',0.5)
title('Rho = -0.7')

subplot(2,1,2)
plot(w2_gauss(:,1), w2_gauss(:,2), 'o','MarkerSize',0.5)
title('Rho = 0.6')

%% Section b)
%Parameters of the bivariate pdf given by (I)
%f(r1,r2) = alfa + beta*r1 + gamma*r2

alfa = 0.0642;
beta = 0.0049;
gamma = 0.0296;

%Conditional sampling
%Conditional distribution from bivar. distrib. = F(r1|r2)

% Marginal cumulative distribution of r2: G(r2)
% G(r2) = G0 + G1*r2 + G2*r2^2

G0 = 0.2222;
G1 = 0.2;
G2 = 0.0444;

% Marginal pdf of r2 = g(r2)
% g(r2) = g00 + g11*r2
% r2 belongs to [-2,2.5]

g00 = 0.2;
g11 = 0.0889;

r2 = zeros(N,4); % draws from r2
r1 = zeros(N,4); % draws from r1

d = alfa -0.5*beta;

copu(:,:,1) = w1_clay;
copu(:,:,2) = w2_clay;
copu(:,:,3) = w1_gauss;
copu(:,:,4) = w2_gauss;

for i=1:N
    for j = 1:4
        G0_aux = G0 - copu(i,2,j);
        A = [G2 G1 G0_aux];
        R = roots(A);
        
        if R(1) < -2
            r2(i,j) = R(2);
        elseif R(1) > 2.5
            r2(i,j) = R(2);
        else
            r2(i,j) = R(1);
        end
    end
end

% r1 by conditional sampling
% r1 belongs to [-1,2]


for i=1:N
    for j = 1:4
        c1 = 1/(g00 + g11*r2(i,j));
        c2 = d + gamma*r2(i,j);
        c3 = alfa + gamma*r2(i,j);
        c4 = 0.5*beta;
        c5 = c2 - U(i,1)/c1;
        B = [c4 c3 c5];
        R = roots(B);
        if R(1) < -1
            r1(i,j) = R(2);
        elseif R(1) > 2
            r1(i,j) = R(2);
        else
            r1(i,j) = R(1);
        end
    end
end

wa = 0.25; % Weight stock 1

RpA = wa*r1 + (1 - wa)*r2;
%correA = corrcoef(r1, r2)
correA = corrcoef(RpA)
mRpMCA = mean(RpA)  %mean
stdRpMCA = std(RpA) %std 

VaR1A = quantile(RpA,0.01) % quantile(1%)
VaR5A = quantile(RpA,0.05) % quantile(5%)
VaR95A = quantile(RpA,0.95) % quantile(95%)
VaR99A = quantile(RpA,0.99) % quantile(99%)

%% Section c)

wpx = [0.5, 0.7, 0.2, 0.15]; % Weights for each portfolio
Rpx = zeros(N, length(wpx));
for i = 1:length(wpx)
    Rpx(:,i) = wpx(i) * r1(:,i) + (1 - wpx(i)) * r2(:,i);
end

correx = corrcoef(Rpx)

% Build Efficient Frontiers (EFs) (short positions allowed)

Nw = 20;

muR = zeros(2, length(Rpx(1,:)));
covR = zeros(2,2,length(Rpx(1,:)));
invcov = zeros(2,2,length(Rpx(1,:)));
l = ones(2,1);
m1 = zeros(2, length(Rpx(1,:)));
m2 = zeros(2, length(Rpx(1,:)));
A = zeros(1, length(Rpx(1,:)));
B = zeros(1, length(Rpx(1,:)));
C = zeros(1, length(Rpx(1,:)));
D = zeros(1, length(Rpx(1,:)));
g = zeros(2, length(Rpx(1,:)));
h = zeros(2, length(Rpx(1,:)));
Wmvp = zeros(2, length(Rpx(1,:)));
Mumvp = zeros(1, length(Rpx(1,:)));
Sigmvp = zeros(1, length(Rpx(1,:)));
Wshort = zeros(Nw,2,length(Rpx(1,:)));
muShort = zeros(Nw,length(Rpx(1,:)));
sigShort = zeros(Nw,length(Rpx(1,:)));
    
for i = 1:length(Rpx(1,:))
    muR(:,i) = [mean(r1(:,i)); mean(r2(:,i))];
    covR(:,:,i) = cov(r1(:,i), r2(:,i));
    invcov(:,:,i) = inv(covR(:,:,i));
    m1(:,i) = invcov(:,:,i)*l;    %vector of order nx1
    m2(:,i) = invcov(:,:,i)*muR(:,i);  %vector of order nx1
    
    A(i) = l'*m2(:,i);        %real number
    B(i) = muR(:,i)'*m2(:,i);      %real number (positive)
    C(i) = l'*m1(:,i);        %real number (positive)
    D(i) = B(i)*C(i)-A(i)^2;      %real number (positive)
    
    g(:,i) = (1/D(i))*(B(i)*m1(:,i)-A(i)*m2(:,i));  %vector of order nx1
    h(:,i) = (1/D(i))*(C(i)*m2(:,i)-A(i)*m1(:,i));  %vector of order nx1
    
    %Min variance EF portfolio (portfolio mvp)
    
    Wmvp(:,i) = (1/C(i))*m1(:,i);       %weight of mvp
    Mumvp(i) = A(i)/C(i);           %mean of mvp
    Sigmvp(i) = sqrt(1/C(i));    %sig of mvp
    
    %Build EF under short position
    
    q = 1.2;
    MuMax1(i) = q*max(muR(:,i));
    mup(i,:) = linspace(Mumvp(i), MuMax1(i), Nw);
   
    for j=1:Nw
        w = g(:,i) + h(:,i)*mup(i,j); %column vector of order length(muR)
        Wshort(j,:,i) = w';
        muShort(j,i) = w'*muR(:,i);
        sigShort(j,i) = sqrt(w'*covR(:,:,i)*w);
    end
end

%Plotting

figure(3);
plot(sigShort(:,1), muShort(:,1),'k')    %Clayton copula
%title('Clayton, \theta = 2')
hold on
plot(sigShort(:,2), muShort(:,2),'r--')  %Clayton copula
%title('Clayton, \theta = 10')
hold on
plot(sigShort(:,3), muShort(:,3),'b')    %Gaussian copula
%title ('Gaussian, \rho = -0.7')
hold on
plot(sigShort(:,4), muShort(:,4),'x')    %Gaussian copula
%title ('Gaussian, \rho = 0.6')
hold on
plot(std(Rpx),mean(Rpx),'d')
legend('Clayton, \theta = 2', 'Clayton, \theta = 10', ...
'Gaussian, \rho = -0.7', 'Gaussian, \rho = 0.6','Location','southwest')
xlabel('portfolio risk')
ylabel('portfolio mean')

newR = zeros(N,4);
for i = 1:length(wpx)
    newR(:,i) = Wmvp(1,i) * r1(:,i) + Wmvp(2,i) * r2(:,i);
end

VaR1 = quantile(newR,0.01) % quantile(1%)
VaR5 = quantile(newR,0.05) % quantile(5%)
