function [full_save_name, utility_user, user_per_k, utility_company, company_per_k]...
    = electricity_userfirst_exponential_r_1(rho, psi_val, phi, lambda, g_def, toplot)

current_path = cd;

%% parameters
if ~exist("phi","var")
    phi = 0.5;
end
if ~exist("rho","var")
    rho = 1.75;
end
if ~exist("psi_val","var")
    psi_val = 10;
end
if ~exist("lambda","var")
    lambda = 5252.94311137;
end
if ~exist("g_def","var")
    g_def = 1;
end
if ~exist("toplot","var")
    toplot = 0;
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

w = psi_val*c;

%plotting config
mkr_load = {'kx-'; 'ko-'; 'k.-'};

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
u1rm=zeros(1,rn); % user distortions for all initializations
u2rm=zeros(1,rn); % company distortions for all initializations

for r=1:rn
    x0=xminit(r,:);

    fun=@(l)f22fn(l,d,alpha,beta,w,lambda,phi,g_def); % objective function

    options = optimoptions('fmincon','MaxFunctionEvaluations',90000000,'MaxIterations',90000000,'Display','off','SpecifyObjectiveGradient',true);
    [l,fval,exitflag,output,lambda_fmin,grad,hessian]=fmincon(fun,x0,[],[],[],[],lmin,lmax,[],options); % gradient descent
    lrm(:,r)=l;

    [p]=pstar(l,w,lambda,phi,g_def);
    prm(:,r)=p;

    s = dissatisfaction(d, beta, l, alpha);
    rk = likelinessOfUser(p,w,phi,l,g_def);

    u1rm(r)=u1(l,p,s);
    u2rm(r)=u2(l,p,c,s,mu,sum(rk),lambda);

    [in1 in2]=min(u1rm(1:r));
    pm=prm(:,in2);
    lm=lrm(:,in2);


    % results
    avg_p           = (pm'*lm)/sum(lm);
    total_load      = sum(lm);
    utility_user    = u1(lm',pm',s)*0.01; %from cents to dollars
    utility_company = u2(lm',pm',c,s,mu,sum(rk),lambda)*0.01; %from cents to dollars
    profit          = companyprofit(c,lm',pm',n,mu)*0.01; %from cents to dollars
    SW              = profit + utility_user;
end %runs

% make from utility to payoff
utility_company     = -utility_company;
utility_user        = -utility_user;

% find l* and compare to load found with fmincon
l_star = [];
for i = 1:length(lm)
    l_star(i,1) = checklstar(w(i),d(i),beta(i),alpha(i),lambda,phi,g_def);
end
l_star = double(l_star);

% table to compare terms of utility functions
p_times_l = pm'*lm;
c_times_l = c*lm;
sum_s = sum(s);
fl = mu * sum((lm-mean(lm)).^2);
termTable = table(rho,p_times_l,c_times_l,sum_s,sum(rk),fl,utility_user,utility_company);

user_per_k                  = u1_perk(lm',pm',s)*0.01;
company_per_k               = u2_perk(lm',pm',c,s,mu,rk,lambda)*0.01;


% ----------------------------------------------------------------------------------------------------------- %
% -------------------------------------------------- Plotting------------------------------------------------ %
% ----------------------------------------------------------------------------------------------------------- %

if toplot
    blocks = 0:n-1;

    %cost and price vs time of day
    f7 = figure;
    stairs(blocks,pm,'k*-','LineWidth',2,'MarkerSize',15)
    hold on
    stairs(blocks,c,'bo-','LineWidth',2,'MarkerSize',15)
    labels_price = {'p', 'c'};
    legend(labels_price)
    ylim([0 max(pm)])
    xlim([0 24])
    grid on
    box on
    ax = gca;
    ax.FontSize = 18;
    ax.GridAlpha = 0.5;
    f7.Position = [100 100 750 400];
    xlabel('k');
    ylabel('Price (cents)');

    
    %load and demand vs time of day
    f8 = figure;
    plot(blocks,lm,mkr_load{1,:},'LineWidth',2,'MarkerSize',15)
    hold on
    plot(blocks,l_star,'gs-','LineWidth',2,'MarkerSize',15)
    hold on
    plot(blocks,d,'bo-','LineWidth',2,'MarkerSize',15)
    hold on
    plot(blocks,lmin,'r^-','LineWidth',2,'MarkerSize',15)
    hold on
    plot(blocks,lmax,'r--','LineWidth',2)
    labels_load = {'$l$' '$l^*$' '$d$' '$\underline{l}$' '$\bar{l}$'};
    legend(labels_load, 'Interpreter','latex','Location','southeast','FontSize',18,'NumColumns',5)
    xlim([0 24])
    grid on
    box on
    ax = gca;
    ax.FontSize = 18;
    ax.GridAlpha = 0.5;
    f8.Position = [100 100 750 400];
    xlabel('k');
    ylabel('Load (GW)');
end

% save data
save_name = strcat('userfirst_lambda',strrep(num2str(lambda),'.','_'),'_psi_val_',strrep(num2str(psi_val),'.','_'),'_exp_r');
results_path = strcat(current_path, "\Results");

if ~exist(results_path)
    mkdir(results_path)
end
full_save_name = strcat(results_path, "\", save_name, ".mat");
save(full_save_name, 'n', 'c', 'd', 'alpha', 'beta', 'lmin','lmax','lm',...
    'pm','profit','utility_company','utility_user','lrm','prm','u1rm','u2rm','xminit','avg_p','total_load',...
    'SW','epsilon','mu','w','termTable','rk','lambda','psi_val','rho','phi','user_per_k','company_per_k')

% save plots
if toplot
    plot_path = strcat(current_path, "\Plots");
    if ~exist(plot_path)
        mkdir(plot_path)
    end
    savefig(f7, strcat(plot_path,"\",save_name, "_PricePlot"));
    savefig(f8, strcat(plot_path,"\",save_name, "_LoadPlot"));
end

fprintf('All Runs Completed\n')
fprintf('Output file: %s\n', save_name)







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
    rk = -exp(phi*p./w); % g(l) = 1
elseif g_def == 2
    rk = -l.^(1/2).*exp(phi*p./w); % g(l) = sqrt(l)
end

function l_star = checklstar(w,d,beta,alpha,lambda,phi,g_def)
syms l_comp
der    = derivativeu1_lk(l_comp,w,d,alpha,beta,lambda,phi,g_def);
l_star = vpasolve(der,l_comp); %solve derivative == 0 for l
l_star = abs(l_star(1,1));

function [dist_dec]=u1_perk(l,p,s) %u1 user
dist_dec = p.*l + s;
dist_dec = -dist_dec; % to make into payoff function

function [dist_enc]=u2_perk(l,p,c,s,mu,rk,lambda) %u2 company
fl       = mu * (l-mean(l)).^2;
dist_enc = - p.*l + c.*l+ s - lambda*rk + fl;
dist_enc = -dist_enc; % to make into payoff function