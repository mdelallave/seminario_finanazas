%% Cleaning 
clear('all')
clc
close('all')

%% Initial parameters
muR = [0.25; 
      0.09;
      0.05];
covR = [ 0.7,   0.1, -0.06;
          0.1,   0.5, -0.1;
         -0.06, -0.1,  0.1];
l = ones(length(muR),1);     
MuMax = max(muR);
Nw = 20;
sigR = diag(diag(sqrt(covR)));
corrR = sigR \ covR / sigR; % corrR = corrcov(covR)

% %% PCTR
% eq_w = l/length(muR); % Equally weighted
% alt_w = [0.2; 
%          0.6;
%          0.2];        % alternative pportfolio
% 
% eqw_sdev = sqrt(eq_w' * covR * eq_w);
% eqw_MCTR = (covR * eq_w) / eqw_sdev;
% eqw_PCTR = eq_w .* eqw_MCTR / eqw_sdev;
% 
% alt_sdev = sqrt(alt_w' * covR * alt_w);
% alt_MCTR = (covR * alt_w) / alt_sdev;
% alt_PCTR = eq_w .* alt_MCTR / alt_sdev;

%% Section e) and m): EF with and without short-selling
invcov = inv(covR);

m1 = invcov*l;    %vector of order nx1
m2 = invcov*muR;  %vector of order nx1

A = l'*m2;        %real number
B = muR'*m2;      %real number (positive)
C = l'*m1;        %real number (positive)
D = B*C-A^2;      %real number (positive)

g = (1/D)*(B*m1-A*m2);  %vector of order nx1
h = (1/D)*(C*m2-A*m1);  %vector of order nx1

%Min variance EF portfolio (portfolio mvp)

Wmvp = (1/C)*m1;       %weight of mvp
Mumvp = A/C;           %mean of mvp
Sigmvp = sqrt(1/C);    %sig of mvp

%Build EF under short position 

q = 1.2; 
MuMax1 = q*MuMax;
mup = linspace(Mumvp,MuMax1,Nw);

Wshort = zeros(Nw,length(muR));
muShort = zeros(Nw,1);
sigShort = zeros(Nw,1);

for i=1:Nw
    w = g + h*mup(i); %column vector of order length(muR)
    Wshort(i,:) = w';
    muShort(i) = w'*muR;
    sigShort(i) = sqrt(w'*covR*w);
end

[PRisk, PRoR, PWts] = NaiveMV(muR, covR, Nw);

WmvpLo = PWts(1,:)';      % Weight of mvp (Long)
MumvpLo = PRoR(1);        % Mean of mvp (Long)
SigmvpLo = PRisk(1);      % Sig of mvp (Long)

% Portfolio comparison: Long vs Short

compaP = [WmvpLo, Wmvp]
compaMV = [MumvpLo, Mumvp; SigmvpLo, SigmvpLo]

% Plotting

figure(1);
plot(PRisk,PRoR) % Long
hold on
plot(sigShort, muShort, 'r--') % Short selling is allowed
hold on
plot(diag(sigR),muR,'d')
xlabel('Portfolio risk')
ylabel('Portfolio mean')

figure(2);
plot(Wshort(:,1), muShort, Wshort(:,2), muShort, Wshort(:,3), muShort) 
xlabel('Weight')
ylabel('\mu_p')
legend('Stock 1', 'Stock 2', 'Stock 3')

%% Section h): quadratic expected utility function

Wd = m2/A;
b = [0.5; 0.3; 0.1];
Wstar = zeros(3);
muP = zeros(1,3);
sigP = zeros(1,3);

for i = 1:3
    Wstar(:,i) = Wmvp + A * (Wd - Wmvp) / b(i);
    muP(i) = Wstar(:,i)'*muR;
    sigP(i) = sqrt(Wstar(:,i)' * covR * Wstar(:,i));
end

Wmixed = [0.6, 0.25, 0.15];
muMixed = Wmixed * muP';
aux = sum(Wmixed .* Wstar, 2);
sigMixed = sqrt(aux' * covR * aux);

% Proof that all of them are efficient (in the PF):
figure(3);
plot(sigShort, muShort) % Short selling is allowed
hold on
plot(sigP,muP,'d', sigMixed,muMixed,'d')
xlabel('Portfolio risk')
ylabel('Portfolio mean')

for i = 1:3
    Wpi(i) = Wmvp(i) + A * (C * muMixed - A) * (Wd(i) - Wmvp(i)) / D;
end

mupi = Wpi * muR; % mupi is equal to muMixed.

%% Section i): risk-free asset:

rf = 0.04;
We = invcov * (muR - rf)/ (l'*invcov * (muR - rf));
muE = We' * muR;
sigE = sqrt(We' * covR * We);

sigCAL = linspace(Mumvp,1,Nw);
muCAL = rf + (muE - rf) * sigCAL/sigE ;

figure(4);
plot(sigShort, muShort) % Short selling is allowed
hold on
plot(sigCAL, muCAL) 
hold on
plot(sigE,muE,'d')
xlabel('Portfolio risk')
ylabel('Portfolio mean')

WtestCAL = [0.15, 0.2, 0.15];
muTestCAL = WtestCAL * muR + (1 - sum(WtestCAL)) * rf;
alpha = (muTestCAL - rf)/(muE - rf);
test = alpha * We;
