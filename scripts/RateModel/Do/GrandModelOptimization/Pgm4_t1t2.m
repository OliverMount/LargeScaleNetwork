load /olive/Maths/R/Do/InputData/myelin_HCP_dk68.mat;
ratio=t1t2Cortex';
ratio=ratio/(max(ratio)-min(ratio));
ratio=ratio-max(ratio)+1;
ratio(find(ratio<0))=0;

fprintf('++ Parameters for the mean field model are set \n')

fprintf('No. of participants considered is %d \n',NSUBSIM);

dtt   = 1e-3;   % Sampling rate of simulated neuronal activity (seconds)
dt=0.1;

taon=100;
taog=10;
gamma=0.641;
sigma=0.01;
JN=0.15;
I0=0.382;
Jexte=1.;
Jexti=0.7;
w=1.4;
 
 
we=2.1;  % This is the brains working point 
alpha=-0.7;
beta=1.4;
gain=1+alpha+(beta*ratio);
Jbal=Balance_J_gain(we,C,gain); % Computed only once before the iteration

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model FC and FCD

kk=1;
for nsub=1:NSUBSIM
    neuro_act=zeros(round(1000*(Tmax+60)*0.754+1),N);
    sn=0.001*ones(N,1);
    sg=0.001*ones(N,1); 
    nn=1;
    for t=0:dt:(1000*(Tmax+60)*0.754)
        xn=I0*Jexte+JN*w*sn+we*JN*C*sn-Jbal.*sg;
        xg=I0*Jexti+JN*sn-sg;
        rn=phie_gain(xn,gain);
        rg=phii_gain(xg,gain);
        sn=sn+dt*(-sn/taon+(1-sn)*gamma.*rn./1000.)+sqrt(dt)*sigma*randn(N,1);
        sn(sn>1) = 1;
        sn(sn<0) = 0;
        sg=sg+dt*(-sg/taog+rg./1000.)+sqrt(dt)*sigma*randn(N,1);
        sg(sg>1) = 1;
        sg(sg<0) = 0;
        j=j+1;
        if abs(mod(t,1))<0.01
            neuro_act(nn,:)=rn';
            nn=nn+1;
        end
    end
    nn=nn-1;
	fprintf('++ Solved rate time series participant - %d  \n',nsub) 

   fprintf('++ Solving BOLD simualted time series for the above participant \n')
    % Friston BALLOON MODEL
    T = nn*dtt; % Total time in seconds
    
    B = BOLD(T,neuro_act(1:nn,1)'); % B=BOLD activity, bf=Foutrier transform, f=frequency range)
    BOLD_act = zeros(length(B),N);
    BOLD_act(:,1) = B;
    
    for nnew=2:N
        B = BOLD(T,neuro_act(1:nn,nnew));
        BOLD_act(:,nnew) = B;
    end
    
    bds=BOLD_act(20:754:end-10,:);
    FC_simul2(nsub,:,:)=corrcoef(bds);
    
    Tmax2=size(bds,1);
    Phase_BOLD_sim=zeros(N,Tmax2);
    BOLDsim=bds';
    for seed=1:N
        BOLDsim(seed,:)=BOLDsim(seed,:)-mean(BOLDsim(seed,:));
        signal_filt_sim =filtfilt(bfilt2,afilt2,BOLDsim(seed,:));
        timeserie(seed,:)=signal_filt_sim;
    end
        %% time decay
        
    fprintf('++ Computing functional connectivity for the participant \n')
    for i=1:N
        for j=1:N
            FCt2(nsub,i,j)=corr2(timeserie(i,1:end-1)',timeserie(j,2:end)');
        end
    end
   
    clear timeserie signal_filt_sim BOLDsim bds Phase_BOLD_sim B BOLD_act neuro_act sn sg rn rg;
	
end 


FC_simul=squeeze(mean(FC_simul2,1));
cc=corrcoef(atanh(FC_emp(Isubdiag)),atanh(FC_simul(Isubdiag)));
FCfitt=cc(2); %% FC fitting

fprintf('Fitted value %f \n', FCfitt);



   fprintf('++ Solving BOLD simualted time series for the above participant \n')
    %%%% BOLD empirical
    % Friston BALLOON MODEL
    T = nn*dtt; % Total time in seconds
    
    B = BOLD(T,neuro_act(1:nn,1)'); % B=BOLD activity, bf=Foutrier transform, f=frequency range)
    BOLD_act = zeros(length(B),N);
    BOLD_act(:,1) = B;
    
    for nnew=2:N
        B = BOLD(T,neuro_act(1:nn,nnew));
        BOLD_act(:,nnew) = B;
    end
    
    bds=BOLD_act(20:754:end-10,:);
    FC_simul2(nsub,:,:)=corrcoef(bds);
    
    Tmax2=size(bds,1);
    Phase_BOLD_sim=zeros(N,Tmax2);
    BOLDsim=bds';
    for seed=1:N
        BOLDsim(seed,:)=BOLDsim(seed,:)-mean(BOLDsim(seed,:));
        signal_filt_sim =filtfilt(bfilt2,afilt2,BOLDsim(seed,:));
        timeserie(seed,:)=signal_filt_sim;
    end
        %% time decay
        
    for i=1:N
        for j=1:N
            FCt2(nsub,i,j)=corr2(timeserie(i,1:end-1)',timeserie(j,2:end)');
        end
    end 
    
 
    
    for i=1:N
        ac=autocorr(timeserie(i,:));
        coef_lin_reg = polyfit(1:5,ac(4:8),1);
        tau(i) = -1/coef_lin_reg(1);
    end
    stausi(nsub,:)=tau*delt;
 

    ii2=1;
    for t=1:18:Tmax2-80
        jj2=1;
        cc=corrcoef((timeserie(:,t:t+80))');
        for t2=1:18:Tmax2-80
            cc2=corrcoef((timeserie(:,t2:t2+80))');
            ca=corrcoef(cc(Isubdiag),cc2(Isubdiag));
            if jj2>ii2
                cotsamplingsim(kk)=ca(2);  %% FCD simulation
                kk=kk+1;
            end
            jj2=jj2+1;
        end
        ii2=ii2+1;
    end
   
    clear timeserie signal_filt_sim BOLDsim bds Phase_BOLD_sim B BOLD_act neuro_act sn sg rn rg;
	
end 


FC_simul=squeeze(mean(FC_simul2,1));
cc=corrcoef(atanh(FC_emp(Isubdiag)),atanh(FC_simul(Isubdiag)));
FCfitt=cc(2); %% FC fitting

fprintf('Static-FC Fitted value %f \n', FCfitt);

[hh pp FCDfitt]=kstest2(cotsamplingdata,cotsamplingsim);  %% FCD fitting

fprintf('Dynamic-FC Fitted value %f \n',1-FCDfitt);


%save("/olive/Maths/R/Do/InputData/FC_simu_filtered.mat","FCt2")
%save("/olive/Maths/R/Do/InputData/FC_simu_nofiltering.mat","FC_simul2")
