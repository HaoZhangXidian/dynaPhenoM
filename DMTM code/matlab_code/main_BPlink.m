clc
clear
close all

%% Load data

load('..\data\diagnosis.mat')
%load('F:\Hao\project\Dynamic_topic_model_for_EHR\python_code\data\Data_diag.mat')
val(val>1)=1;
X_diagnosis = sparse(double(row+1),double(col+1),val,double(num_row),double(num_col));
X_diagnosis_binary = X_diagnosis;

[ii_diagnosis,jj_diagnosis] =   find(X_diagnosis>eps);    %Treat values smaller than eps as 0
iijj_diagnosis    =   find(X_diagnosis>eps);

load('..\data\procedure.mat')
val(val>1)=1;
X_procedure = sparse(double(row+1),double(col+1),val,double(num_row),double(num_col));
X_procedure_binary = X_procedure;

[ii_procedure,jj_procedure] =   find(X_procedure>eps);    %Treat values smaller than eps as 0
iijj_procedure    =   find(X_procedure>eps);

load('..\data\drug.mat')
val(val>1)=1;
X_prescription = sparse(double(row+1),double(col+1),val,double(num_row),double(num_col));
X_prescription_binary = X_prescription;

[ii_prescription,jj_prescription] =   find(X_prescription>eps);    %Treat values smaller than eps as 0
iijj_prescription   =   find(X_prescription>eps);

Patient_label = Patient_label+1;

%% parameter
L = 1;
K = 50;
V_diagnosis = size(X_diagnosis,1);
V_prescription = size(X_prescription,1);
V_procedure = size(X_procedure,1);
N = size(X_diagnosis,2);
iteration = 500;
num_patient = max(Patient_label);

burn_in = 200;
step = 5;

%% Hyperparameter
Supara.tao0 = 1; % parameter in transition gamma distribution
Supara.eta0 = 0.01; % Prior of Phi
Supara.gamma0 = 100; % prior for pi
Supara.epilson0 = 0.1;
%% Initialization
Para.Phi_diagnosis = rand(V_diagnosis,K);
Para.Phi_diagnosis = bsxfun(@rdivide, Para.Phi_diagnosis, max(realmin,sum(Para.Phi_diagnosis,1)));
Para.Phi_prescription = rand(V_prescription,K);
Para.Phi_prescription = bsxfun(@rdivide, Para.Phi_prescription, max(realmin,sum(Para.Phi_prescription,1)));
Para.Phi_procedure = rand(V_procedure,K);
Para.Phi_procedure = bsxfun(@rdivide, Para.Phi_procedure, max(realmin,sum(Para.Phi_procedure,1)));

Para.Pi  = eye(K);
Para.Xi = 1;
Para.V = ones(K,1);
Para.h = zeros(K,K);
Para.beta = 1;
Para.q = ones(K,1);
Para.n = ones(K,1);
Para.rou = ones(K,1);

Theta    = ones(K,N)/K;

delta_diagnosis = ones(N,1);
delta_prescription = ones(N,1);
delta_procedure = ones(N,1);

L_dotkt_all = cell(num_patient,1);

Likelihood_Total = zeros(iteration,1);
Likelihood_Diag = zeros(iteration,1);
Likelihood_Drug = zeros(iteration,1);
Likelihood_Proc = zeros(iteration,1);

delta_diagnosis_sum = 0;
delta_prescription_sum = 0;
delta_procedure_sum = 0;

Phi_diagnosis_sum = 0;
Phi_prescription_sum = 0;
Phi_procedure_sum = 0;
Pi_sum = 0;
Theta_sum = 0;
flag = 0;

for i=1:iteration
    tic
    Rate =  Para.Phi_diagnosis*bsxfun(@times,delta_diagnosis', Theta);
    M = truncated_Poisson_rnd(Rate(iijj_diagnosis));
    clear Rate
    X_diagnosis = sparse(ii_diagnosis,jj_diagnosis,M,V_diagnosis,N);
    
    Rate =  Para.Phi_procedure*bsxfun(@times,delta_procedure', Theta);
    M = truncated_Poisson_rnd(Rate(iijj_procedure));
    clear Rate
    X_procedure = sparse(ii_procedure,jj_procedure,M,V_procedure,N);
    
    Rate =  Para.Phi_prescription*bsxfun(@times,delta_prescription', Theta);
    M = truncated_Poisson_rnd(Rate(iijj_prescription));
    clear Rate
    X_prescription = sparse(ii_prescription,jj_prescription,M,V_prescription,N);
    
    [A_KN_diagnosis,A_VK_diagnosis] = Multrnd_Matrix_mex_fast_v1(X_diagnosis, Para.Phi_diagnosis, bsxfun(@times,delta_diagnosis', Theta));
    [A_KN_prescription,A_VK_prescription] = Multrnd_Matrix_mex_fast_v1(X_prescription, Para.Phi_prescription, bsxfun(@times,delta_prescription', Theta));
    [A_KN_procedure,A_VK_procedure] = Multrnd_Matrix_mex_fast_v1(X_procedure, Para.Phi_procedure, bsxfun(@times,delta_procedure', Theta));
    
    L_KK = zeros(K,K);  % For update Pi  Eq(25)
    L_kdott_for_V = zeros(K,num_patient); % For update V
    
    %% sample latent counts
    for n=1:num_patient
        x_patient_diagnosis = full(X_diagnosis(:, Patient_label==n));
        x_patient_prescription = full(X_prescription(:, Patient_label==n));
        x_patient_procedure = full(X_procedure(:, Patient_label==n));
        
        patinet_T = size(x_patient_diagnosis,2);
        
        L_kdott = zeros(K, patinet_T);
        
        Theta_patient = Theta(:, Patient_label==n);
        A_KN_patient = A_KN_diagnosis(:, Patient_label==n) + A_KN_prescription(:, Patient_label==n) + A_KN_procedure(:, Patient_label==n);
        
        if i==1
            L_dotkt_all{n} = zeros(K,patinet_T+1);
        end
        
        for t=patinet_T:-1:2
            L_kdott(:,t) = CRT_sum_mex_matrix_v1(sparse((A_KN_patient(:,t)+ L_dotkt_all{n}(:,t+1))'),(Supara.tao0 * Para.Pi * Theta_patient(:,t-1))')'; % Eq (12), (16), (20)
            [L_dotkt_all{n}(:,t),tmp] = Multrnd_Matrix_mex_fast_v1(sparse(L_kdott(:,t)), Para.Pi,Theta_patient(:,t-1));
            L_KK = L_KK + tmp;
        end
        
    end
    
    %% sample Global
    Para.Phi_diagnosis = SamplePhi(A_VK_diagnosis,Supara.eta0);
    if nnz(isnan(Para.Phi_diagnosis))
        warning('Diagnosis Phi Nan');
        Para.Phi_diagnosis(isnan(Para.Phi_diagnosis)) = 0;
    end
    
    Para.Phi_prescription = SamplePhi(A_VK_prescription,Supara.eta0);
    if nnz(isnan(Para.Phi_prescription))
        warning('Prescription Phi Nan');
        Para.Phi_prescription(isnan(Para.Phi_prescription)) = 0;
    end
    
    Para.Phi_procedure = SamplePhi(A_VK_procedure,Supara.eta0);
    if nnz(isnan(Para.Phi_procedure))
        warning('Procedure Phi Nan');
        Para.Phi_procedure(isnan(Para.Phi_procedure)) = 0;
    end
    
    Piprior = Para.V*Para.V';
    Piprior(logical(eye(size(Piprior))))=0;
    Piprior = Piprior+diag(Para.Xi*Para.V);
    Para.Pi = SamplePi(L_KK,Piprior);
    if nnz(isnan(Para.Pi))
        warning('Pi Nan');
        Para.Pi(isnan(Para.Pi)) = 0;
    end
    
    %% calculate Local
    L_dotkt_all_sum = 0;
    Zeta_all_for_v = 0;
    for n=1:num_patient
        x_patient_diagnosis = X_diagnosis(:, Patient_label==n);
        patinet_T = size(x_patient_diagnosis,2);
        Theta_patient = Theta(:, Patient_label==n);
        A_KN_patient = A_KN_diagnosis(:, Patient_label==n) + A_KN_prescription(:, Patient_label==n) + A_KN_procedure(:, Patient_label==n);
        delta_patient = delta_diagnosis(Patient_label==n) + delta_prescription(Patient_label==n) + delta_procedure(Patient_label==n);
        Zeta_patient = zeros(patinet_T+1,1);
        
        for t=patinet_T:-1:1
            Zeta_patient(t) = real(log(1 + delta_patient(t)/Supara.tao0 + Zeta_patient(t+1)));
        end
        
        Zeta_all_for_v = Zeta_all_for_v + Zeta_patient(1);
        L_dotkt_all_sum = L_dotkt_all_sum+sum(L_dotkt_all{n},2);
        
        for t=1:patinet_T
            if t==1
                shape = A_KN_patient(:,t)+ L_dotkt_all{n}(:,t+1)+ Supara.tao0 * Para.V;
            else
                shape = A_KN_patient(:,t)+ L_dotkt_all{n}(:,t+1)+ Supara.tao0*(Para.Pi* Theta_patient(:,t-1));
            end
            scale = Supara.tao0 + delta_patient(t)+ Zeta_patient(t+1);
            Theta_patient(:,t) = gamrnd(shape,1./scale);
        end
        if nnz(isnan(Theta_patient))
            warning('Theta Nan');
        end
        Theta(:, Patient_label==n) = Theta_patient;
    end
    
    %% Others
    shape = Supara.epilson0 + Supara.gamma0;
    scale = Supara.epilson0 + sum(Para.V);
    Para.beta = gamrnd(shape,1./scale);
    
    a  = L_dotkt_all_sum;
    b  = Para.V.*(Para.Xi+repmat(sum(Para.V),K,1)-Para.V);
    Para.q = betarnd(b,a);
    Para.q = max(Para.q,realmin);
    
    for k1 = 1:K
        for k2 = 1:K
            Para.h(k1,k2) = CRT_sum_mex_matrix_v1(sparse(L_KK(k1,k2)),Piprior(k1,k2));
        end
    end
    shape = Supara.gamma0/K + trace(Para.h);
    scale = Para.beta - Para.V'*log(Para.q);
    Para.Xi = gamrnd(shape,1./scale);
    
    for n=1:num_patient
        A_KN_patient = A_KN_diagnosis(:, Patient_label==n) + A_KN_prescription(:, Patient_label==n) + A_KN_procedure(:, Patient_label==n);
        for k=1:K
            L_kdott_for_V(k,n) = CRT_sum_mex_matrix_v1(sparse(A_KN_patient(k,1)+ L_dotkt_all{n}(k,2)),Supara.tao0*Para.V(k));
        end
    end
    for k=1:K
        Para.n(k)=sum(Para.h(k,:)+Para.h(:,k)')-Para.h(k,k) + sum(L_kdott_for_V(k,:),2);
        Para.rou(k) = -log(Para.q(k)) * (Para.Xi+sum(Para.V)-Para.V(k)) - (log(Para.q')*Para.V-log(Para.q(k))*Para.V(k)) + Zeta_all_for_v;
    end
    shape = Supara.gamma0/K + Para.n;
    scale = Para.beta + Para.rou;
    Para.V = gamrnd(shape,1./scale);
    
    for n=1:num_patient
        x_patient_diagnosis = full(X_diagnosis(:, Patient_label==n));
        x_patient_prescription = full(X_prescription(:, Patient_label==n));
        x_patient_procedure = full(X_procedure(:, Patient_label==n));
        Theta_patient = Theta(:, Patient_label==n);
        
        delta_patient_diagnosis = Sample_delta(x_patient_diagnosis,Theta_patient,Supara.epilson0,0);
        delta_patient_prescription = Sample_delta(x_patient_prescription,Theta_patient,Supara.epilson0,0);
        delta_patient_procedure = Sample_delta(x_patient_procedure,Theta_patient,Supara.epilson0,0);
        
        delta_diagnosis(Patient_label==n) = delta_patient_diagnosis;
        delta_prescription(Patient_label==n) = delta_patient_prescription;
        delta_procedure(Patient_label==n) = delta_patient_procedure;
    end
    
    if nnz(isnan(delta_diagnosis))
        warning('Diagnosis delta Nan');
    end
    if nnz(isnan(delta_prescription))
        warning('Prescription delta Nan');
    end
    if nnz(isnan(delta_procedure))
        warning('Procedure delta Nan');
    end
    
    %% Calculate likelihood
    
    Lambda = Para.Phi_diagnosis * bsxfun(@times,delta_diagnosis', Theta);
    P = 1-exp(-Lambda);
    P(P==0) = eps;
    P(P==1) = 1-eps;
    Likelihood_diagnosis = sum(sum(X_diagnosis_binary.*log(P) + (1-X_diagnosis_binary).*log(1-P)))/V_diagnosis;    

    toc
    Lambda = Para.Phi_prescription * bsxfun(@times,delta_prescription', Theta);
    P = 1-exp(-Lambda);
    P(P==0) = eps;
    P(P==1) = 1-eps;
    Likelihood_prescription = sum(sum(X_prescription_binary.*log(P) + (1-X_prescription_binary).*log(1-P)))/V_prescription;
    
    Lambda = Para.Phi_procedure * bsxfun(@times,delta_procedure', Theta);
    P = 1-exp(-Lambda);
    P(P==0) = eps;
    P(P==1) = 1-eps;
    Likelihood_procedure = sum(sum(X_procedure_binary.*log(P) + (1-X_procedure_binary).*log(1-P)))/V_procedure;
    
    Likelihood_all = Likelihood_diagnosis + Likelihood_prescription + Likelihood_procedure;
    
    Likelihood_Total(i) = Likelihood_all;
    Likelihood_Diag(i) = Likelihood_diagnosis;
    Likelihood_Drug(i) = Likelihood_prescription;
    Likelihood_Proc(i) = Likelihood_procedure;
    
    fprintf('Iteration %d/%d, All Likelihood %f, Diagnosis Likelihood %f, Prescription Likelihood %f, Procedure Likelihood %f \n',i, iteration, Likelihood_all, Likelihood_diagnosis, Likelihood_prescription, Likelihood_procedure);
    %%
    if i>burn_in
        if mod(i, step) == 0
            delta_diagnosis_sum = delta_diagnosis_sum + delta_diagnosis;
            delta_prescription_sum = delta_prescription_sum + delta_prescription;
            delta_procedure_sum = delta_procedure_sum + delta_procedure;
            
            Phi_diagnosis_sum = Phi_diagnosis_sum +  Para.Phi_diagnosis;
            Phi_prescription_sum = Phi_prescription_sum +  Para.Phi_prescription;
            Phi_procedure_sum = Phi_procedure_sum +  Para.Phi_procedure;
            
            Theta_sum = Theta_sum + Theta;
            Pi_sum = Pi_sum + Para.Pi;
            flag = flag + 1;
            
            Phi_diagnosis_mean = Phi_diagnosis_sum / flag;
            Phi_prescription_mean = Phi_prescription_sum / flag;
            Phi_procedure_mean = Phi_procedure_sum / flag;
            
            delta_diagnosis_mean = delta_diagnosis_sum / flag;
            delta_prescription_mean = delta_prescription_sum / flag;
            delta_procedure_mean = delta_procedure_sum / flag;
            
            Theta_mean = Theta_sum / flag;
            Pi_mean = Pi_sum / flag;
            
            Lambda = Phi_diagnosis_mean * bsxfun(@times,delta_diagnosis_mean', Theta_mean);
            P = 1-exp(-Lambda);
            P(P==0) = eps;
            P(P==1) = 1-eps;
            Likelihood_diagnosis = sum(sum(X_diagnosis_binary.*log(P) + (1-X_diagnosis_binary).*log(1-P)))/V_diagnosis;
            
            Lambda = Phi_prescription_mean * bsxfun(@times,delta_prescription_mean', Theta_mean);
            P = 1-exp(-Lambda);
            P(P==0) = eps;
            P(P==1) = 1-eps;
            Likelihood_prescription = sum(sum(X_prescription_binary.*log(P) + (1-X_prescription_binary).*log(1-P)))/V_prescription;
            
            Lambda = Phi_procedure_mean * bsxfun(@times,delta_procedure_mean', Theta_mean);
            P = 1-exp(-Lambda);
            P(P==0) = eps;
            P(P==1) = 1-eps;
            Likelihood_procedure = sum(sum(X_procedure_binary.*log(P) + (1-X_procedure_binary).*log(1-P)))/V_procedure;

            fprintf('Iteration %d/%d, sum All Likelihood %f, Diagnosis Likelihood %f, Prescription Likelihood %f, Procedure Likelihood %f \n',i, iteration, Likelihood_all, Likelihood_diagnosis, Likelihood_prescription, Likelihood_procedure);
        end
    end
    
end

Theta_mean = Theta_sum / flag;
Pi_mean = Pi_sum / flag;
Phi_diagnosis_mean = Phi_diagnosis_sum / flag;
Phi_prescription_mean = Phi_prescription_sum / flag;
Phi_procedure_mean = Phi_procedure_sum / flag;

Phi_diagnosis = Para.Phi_diagnosis;
Phi_prescription = Para.Phi_prescription;
Phi_procedure = Para.Phi_procedure;
Pi = Para.Pi;

save('.\trained_model\mm_MarketScan.mat','Phi_diagnosis','Phi_prescription','Phi_procedure','Phi_diagnosis_mean','Phi_prescription_mean','Phi_procedure_mean','Pi_mean','Theta_mean')


