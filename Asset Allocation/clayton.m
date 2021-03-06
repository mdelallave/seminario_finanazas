%clayton.m
%Simulate dep. U(0,1) from Clayton copula
%function u=clayton(alfa,epsi)
%alfa=Clayton parameter
%epsi=matrix (n1,n2) of indep. U(0,1) 


function u=clayton(alfa,epsi)

[n1,n2]=size(epsi);

u=zeros(n1,n2);  %matrix of dep. U(0,1)


w2=-alfa/(alfa+1);

w3=-alfa/(2*alfa+1);


u(:,1)=epsi(:,1);

u(:,2)=((u(:,1).^(-alfa)).*(epsi(:,2).^w2 - 1)+1).^(-1/alfa);

%u(:,3)=((u(:,1).^(-alfa)+u(:,2).^(-alfa)-1).*(epsi(:,3).^w3 - 1)+1).^(-1/alfa);




