%simuN.m
%Simulate of dep. U(0,1) from Gaussian copula
%function u=simuN(corre,epsi)
%corre = correlation matrix   
%epsi=matrix (n1,n2) of indep. U(0,1)


function u=simuN(corre,epsi)

 [n1,n2]=size(epsi);

 z=zeros(n1,n2);

 epsi=norminv(epsi,0,1); %matrix of indep. N(0,1)

 B=chol(corre,'lower'); %Cholesky decomposition
  
 z(:,1)=epsi(:,1);
 
 z(:,2)=epsi*B(2,:)';
 
 %z(:,3)=epsi*B(3,:)';
 
 
 u = normcdf(z,0,1); %matrix of dep. U(0,1) 
   
 
 
 
 
 
 