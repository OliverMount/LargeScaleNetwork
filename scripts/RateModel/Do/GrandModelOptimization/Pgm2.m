disp('++ Parameters for the mean field model are set')
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
 
 
we=2.1;
%gain=1+alpha+beta*ratio;
gain=1;  % For homogenoue model gain is 1  (as no heterogenity introduced)
Jbal=Balance_J_gain(we,C,gain); % Computed only once before the iteration

% Model FC and FCD

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
   



   disp("++ Solving BOLD simualted time series for the above participant")
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
        
    disp("++ Computing functional connectivity for the participant")
    for i=1:N
        for j=1:N
            FCt2(nsub,i,j)=corr2(timeserie(i,1:end-1)',timeserie(j,2:end)');
        end
    end
   
    clear timeserie signal_filt_sim BOLDsim bds Phase_BOLD_sim B BOLD_act neuro_act sn sg rn rg;
	
 end 

save("/olive/Maths/R/Do/InputData/FC_simu_filtered.mat","FCt2")
save("/olive/Maths/R/Do/InputData/FC_simu_nofiltering.mat","FC_simul2")
