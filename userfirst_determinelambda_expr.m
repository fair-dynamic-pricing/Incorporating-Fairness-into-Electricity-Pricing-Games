function lambda = userfirst_determinelambda_expr(rho, psi, phi, g_def)

current_path = cd;

%% parameters
if ~exist("rho","var")
    rho = 1.75;
end
if ~exist("psi","var")
    psi = 10;
end
if ~exist("phi","var")
    phi = 0.5;
end
if ~exist("g_def","var")
    g_def = 1;
end


c           = [2.4 2.3 2.3 2.3 2.3 2.4 2.5 2.9 3 3.7 4.9 4.9 5.1 6 7.3 8 7.5 5.5 4.9 4.6 4.9 4.6 2.7 2.6]; % marginal cost (0-24 hrs)
d           = [52 52 49 50 51 54 60 66 70 73 76 79 82 83 86 86 86 85 83 82 81 79 71 64.1]; % nominal demand
lmin        = [48 47 46 47 48 50 56 60 63 66 69 71 74 75 77 78 78 77 75 73 73 70 64 58]; % lower limit of load   3 * ones(1,length(c)); %
lmax        = [79 76 74 74 77 81 92 95 95 95 95 95 95 95 95 95 95 95 95 95 95 95 95 95]; % upper limmer of load

n           = length(c);
rn          = 100; %runs

epsilon     = -[.5 .4 .35 .34 .33 .34 .4 .52 .35 .3 .28 .25 .88 .88 .52 .24 .26 .38 .38 .35 .4 .45 .55 .51];%1 /(alpha(1) - 1); %between -1.29 and -0.2 in paper. less than 0
alpha       = ones(1,n) ./ epsilon+1 ; %less than 1
eta         = 10*ones(1,n); %10 in paper. greater than 0
beta        = -eta./alpha; % alpha * beta < 0
mu          = 1;

r0  = n*(exp(phi*rho/psi)) 
tol = 1e-5;

w = psi*c;

% initializations
xminit = zeros(rn,n);
for n1=1:rn
    for i = 1:n
        xminit(n1,i) =  lmin(i)+(lmax(i)-lmin(i))*rand(1,1);
    end
end
rn=size(xminit,1); % number of initializations

lrm=zeros(n,rn); % final quantizer values for all initializations
prm=zeros(n,rn); % final quantizer representative values for all initializations
lambda = 100;
lambdaflag = 0;

while lambdaflag == 0
    for r=1:rn
        x0=xminit(r,:);

        fun=@(l)f22fn(l,d,alpha,beta,w,lambda,phi,g_def); % objective function

        options = optimoptions('fmincon','MaxFunctionEvaluations',90000000,'MaxIterations',90000000,'Display','off','SpecifyObjectiveGradient',true);%,'CheckGradients',true);
        % ,'Display','none');
        [l,fval,exitflag,output,lambda_fmin,grad,hessian]=fmincon(fun,x0,[],[],[],[],lmin,lmax,[],options); % gradient descent
        lrm(:,r)=l;

        [p]=pstar(l,w,lambda,phi,g_def);
        prm(:,r)=p;

        rk = likelinessOfUser(p,w,phi,l,g_def);
    end %run
    rk = -rk

    %check if lambda passes
    if abs((r0-rk)/r0) < tol
        lambdaflag = 1;
    else 
        multiplier = rk/r0;
        lambda = lambda*multiplier;
    end
end %while

fprintf('Lambda value found.\n')
fprintf('Lambda value = %0.8f for r0 = %0.4f \n',lambda,r0)


save_name = strcat('userfirst_rho',strrep(num2str(rho),'.','_'),'_psi',strrep(num2str(psi),'.','_'),'_fairness2_phi',strrep(num2str(phi),'.','_'));
results_path = strcat(current_path, "\Results\determinelambda\");
if ~exist(results_path)
    mkdir(results_path)
end
full_save_name = strcat(results_path, "\", save_name, ".mat");
save(full_save_name, 'lambda')





% ----------------------------------------------------------------------------------------------------------- %
% ------------------------------------------- Supporting Functions ---------------------------------------- %
% ----------------------------------------------------------------------------------------------------------- %



function [der]=derivativeu1_lk(l,w,d,alpha,beta,lambda,phi,g_def)
if g_def == 1
    der     = w/phi + (w.*log((l.*w)/(lambda*phi)))/phi + alpha.*beta.*((l./d).^(alpha-1)); % g(l) = 1
elseif g_def ==2
    der     = w/(2*phi) + (w.*log((l.^(1/2).*w)/(lambda*phi)))/phi + alpha.*beta.*((l./d).^(alpha-1)); % g(l) = sqrt(l)
end

function [dist_dec]=u1(l,p,s) %u1 user
dist_dec = p*l'+ sum(s);

function [dist_enc]=u2(l,p,c,s,mu,rk,lambda) %u2 company
fl       = mu * sum((l-mean(l)).^2);
dist_enc = - p*l' + c*l'+ sum(s) - lambda*rk + fl;

function [dist_dec]=companyprofit(c,l,p,n,mu)
fl       = mu * sum((l-(sum(l)/n)).^2);
dist_dec = p*l'-c*l' - fl;

function [p]=pstar(l,w,lambda,phi,g_def)
% take the derivative of u2 and solves for p to find the maximum p
if g_def == 1
    p       = (w.*log((l.*w)/(lambda*phi)))/phi; % g(l) = 1
elseif g_def == 2
    p       = (w.*log((l.^(1/2).*w)/(lambda*phi)))/phi; % g(l) = sqrt(l)
end

function [fun,der]=f22fn(l,d,alpha,beta,w,lambda,phi,g_def)
[p]     = pstar(l,w,lambda,phi,g_def);
s       = dissatisfaction(d, beta, l, alpha);
fun     = u1(l,p,s);
[der]   = derivativeu1_lk(l,w,d,alpha,beta,lambda,phi,g_def);


function s = dissatisfaction(d, beta, l, alpha)
s = d.*beta.*((l./d).^alpha - 1);

function rk = likelinessOfUser(p,w,phi,l,g_def)
if g_def == 1
    rk = sum(-exp(phi*p./w)); % g(l) = 1
elseif g_def == 2
    rk = sum(-l.^(1/2).*exp(phi*p./w)); % g(l) = sqrt(l)
end
