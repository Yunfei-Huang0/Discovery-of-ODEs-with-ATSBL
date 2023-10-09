clear all, close all, clc
% Code by Yunfei Huang
% For Paper, "Sparse inference and active
% learning of stochastic diferential equations from data"
% by Y. Huang, Y. Mabrouk, G. Gompper, B. Sabass

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Genreate Data for the  Lorenz system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This part code is exact the same as the code by Steven L. Brunton
% For Paper, "Discovering Governing Equations from Data: 
%        Sparse Identification of Nonlinear Dynamical Systems"
% by S. L. Brunton, J. L. Proctor, and J. N. Kutz
sigma = 10;  
beta  = 8/3;
rho   = 28;
n     = 3;
x0    =[-8; 8; 27];

dt    = 0.0002;
tspan = [.001:dt:10];
N     = length(tspan);
options = odeset('RelTol',1e-12,'AbsTol',1e-12*ones(1,n));
[t,xx]   = ode45(@(t,x) lorenz(t,x,sigma,beta,rho),tspan,x0,options);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate time derivative
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f_size = size(xx,1)-4;
for t = 1: f_size
    zt(t,1) =(-xx(t+4,3) +8.*xx(t+3,3) -8.*xx(t+1,3) + xx(t,3))./(12*dt);
end

x = xx(3:end-2,1);
y = xx(3:end-2,2);
z = xx(3:end-2,3);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The library matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I=ones(size(x,1),1);
     
Theta = [ x           y                z               x.^2        x.*y        x.*z    ...
          y.^2         y.*z            z.^2             x.^3        x.^2.*y     x.^2.*z    x.*y.^2 ...
          x.*y.*z      x.*z.^2         y.^3             y.^2.*z     y.*z.^2     z.^3 ... 
          x.^4         x.^3.*y         x.^3.*z          x.^2.*y.^2  x.^2.*y.*z    ...
          x.^2.*z.^2   x.*y.^3         x.*y.^2.*z       x.*y.*z.^2   x.*z.^3     y.^4 ...
          y.^3.*z      y.^2.*z.^2      y.*z.^3          z.^4         x.^5    x.^4.*y ...
          x.^4.*z      x.^3.*y.^2      x.^3.*y.*z        x.^3.*z.^2    ...
          x.^2.*y.^3   x.^2.*y.^2.*z   x.^2.*y.*z.^2     x.^2.*z.^3 ...
          x.*y.^4      x.*y.^3.*z      x.*y.^2.*z.^2     x.*y.*z.^3 ...
          x.*z.^4      y.^5            y.^4.*z ...
          y.^3.*z.^2   y.^2.*z.^2      y.*z.^4   z.^5];
   
 Xi = [ "I"   "x"            "y"               "z"               "x.^2"         "x.*y"         "x.*z"               ...
              "y.^2"         "y.*z"            "z.^2"            "x.^3"         "x.^2.*y"      "x.^2.*z"    "x.*y.^2" ...
              "x.*y.*z"      "x.*z.^2"         "y.^3"            "y.^2.*z"      "y.*z.^2"      "z.^3"                ... 
              "x.^4"         "x.^3.*y"         "x.^3.*z"         "x.^2.*y.^2"   "x.^2.*y.*z"                       ...
              "x.^2.*z.^2"   "x.*y.^3"         "x.*y.^2.*z"      "x.*y.*z.^2"   "x.*z.^3"      "y.^4"                      ...
              "y.^3.*z"      "y.^2.*z.^2"      "y.*z.^3"         "z.^4"         "x.^5"          "x.^4.*y"            ...
              "x.^4.*z"      "x.^3.*y.^2"      "x.^3.*y.*z"      "x.^3.*z.^2"    ...
              "x.^2.*y.^3"   "x.^2.*y.^2.*z"   "x.^2.*y.*z.^2"   "x.^2.*z.^3" ...
              "x.*y.^4"      "x.*y.^3.*z"      "x.*y.^2.*z.^2"   "x.*y.*z.^3" ...
              "x.*z.^4"      "y.^5"            "y.^4.*z" ...
              "y.^3.*z.^2"   "y.^2.*z.^2"      "y.*z.^4"   "z.^5"]';   
    
Theta =[I Theta];


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ATSBL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n_t = fix(0.80 * size(zt,1));
 
Theta_train = Theta(1:n_t,:);
Theta_test = Theta(n_t+1:end,:);
zt_train = zt(1:n_t,1);
zt_test = zt(n_t+1:end,:);
 
dtol = 5e-3;  
n = 1e-3*cond(Theta);
tol = dtol; 
initsigma2 = std(zt_train)^2/1e2;
lambdap = [];
tol_iters = 30;
T_y = size(Theta_train, 2);

xi_best = Theta_train \ zt_train;
error_best = (norm(Theta_test*xi_best-zt_test,2)) + n*sum(abs(xi_best)>0);

for i=1:tol_iters
   
   [weights,used, s1, e1] = FastLaplace(Theta_train,zt_train,initsigma2,1e-8,lambdap);
   nubnode=size(Theta_train);
   F1 = zeros(nubnode(2),1);
   F1(used) = weights; 

   big1 = find(abs(F1) >= tol);
   small1 = find(abs(F1) < tol);
   
   Theta_train_old = Theta_train;
   Theta_train(:, small1) = [];

   if i ==1
       fin_position = big1; 
       xi = F1;
   else 
       fin_position = fin_position(big1);
       xi = zeros(T_y,1);
       xi(fin_position) = F1;
   end 
   
   error = (norm(Theta_test * xi - zt_test,2)) + n*sum(abs(xi)>0);
   
  if error <= error_best 
      error_best = error;
      xi_best = xi;
      tol = tol + dtol;
      
  else
      Theta_train = Theta_train_old;
      tol = max(0, tol-2*dtol);
      dtol = 2*dtol/(tol_iters - i);
      tol = tol+dtol;
  end
  
end

result=[Xi xi_best]; 

clearvars -except result



