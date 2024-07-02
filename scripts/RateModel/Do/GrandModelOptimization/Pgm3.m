
fprintf('No. of participants considered is %d \n',NSUBSIM);

 
for nsub=1:NSUBSIM
    tsdata(:,:,nsub)=ts(:,[1:34 42:75],nsub);
    FCdata(nsub,:,:)=corrcoef(squeeze(tsdata(:,:,nsub)));
end

FC_emp=squeeze(mean(FCdata,1));

FCemp2=FC_emp-FC_emp.*eye(N);
GBCemp=mean(FCemp2,2);

Isubdiag = find(tril(ones(N),-1));

fprintf('++ Dealing with the orignal data fitting  \n');

kk=1;
for nsub=1:NSUBSIM
    BOLDdata=(squeeze(tsdata(:,:,nsub)))';
    for seed=1:N
        BOLDdata(seed,:)=BOLDdata(seed,:)-mean(BOLDdata(seed,:));
        timeseriedata(seed,:) =filtfilt(bfilt2,afilt2,BOLDdata(seed,:));
    end


    %% time decay
    for i=1:N
        for j=1:N
            FCtdata2(nsub,i,j)=corr2(timeseriedata(i,1:end-1)',timeseriedata(j,2:end)');
        end
    end 

    for i=1:N
        ac=autocorr(timeseriedata(i,:));
        coef_lin_reg = polyfit(1:5,ac(4:8),1);
        tau(i) = -1/coef_lin_reg(1);
    end
    stau(nsub,:)=tau*delt;
    %%

    ii2=1;
    for t=1:18:Tmax-80
        jj2=1;
        cc=corrcoef((timeseriedata(:,t:t+80))');
        for t2=1:18:Tmax-80
            cc2=corrcoef((timeseriedata(:,t2:t2+80))');
            ca=corrcoef(cc(Isubdiag),cc2(Isubdiag));
            if jj2>ii2
        cotsamplingdata(kk)=ca(2);   %% this accumulate all elements of the FCD empirical
                kk=kk+1;
            end
            jj2=jj2+1;
        end
        ii2=ii2+1;
    end
end
FCtdata=squeeze(nanmean(FCtdata2,1));
staudata=nanmean(stau,1); 


fprintf('++ Parameters for the mean field model are set \n')


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
alpha=0.3;
beta=1.8;
gain=1+alpha+(beta*ratio);
Jbal=Balance_J_gain(we,C,gain); % Computed only once before the iteration

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Model FC and FCD

kk=1;
TS_LEN=length(0:dt:(1000*(Tmax+60)*0.754));
idx=1:TS_LEN;
 

 
for nsub=1:NSUBSIM
    neuro_act=zeros(round(1000*(Tmax+60)*0.754+1),N);
    
    xn=zeros(N,TS_LEN);
    xg=zeros(N,TS_LEN);
	   
    sn=0.001*ones(N,1);
    sg=0.001*ones(N,1);  

    nn=1;
    tt=1;
    for t=0:dt:(1000*(Tmax+60)*0.754)
        xn(:,tt)=I0*Jexte+JN*w*sn+we*JN*C*sn-Jbal.*sg;
        xg(:,tt)=I0*Jexti+JN*sn-sg;
        rn=phie_gain(xn(:,tt),gain);
        rg=phii_gain(xg(:,tt),gain);
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
	tt=tt+1;
    end
    nn=nn-1;
    	
    fprintf('++ Obtained current time series for participant - %d  \n',nsub); 
    fprintf('++ Saving the current time series for participant - %d  \n',nsub); 
	
    str_exc= "/olive/Maths/R/Do/OutputData/IExi_" + num2str(nsub) + ".mat";
    str_inh= "/olive/Maths/R/Do/OutputData/IInh_" + num2str(nsub) + ".mat";

    save(str_exc,'xn','-v7.3'); % saving the excitatory current
    save(str_inh,'xg','-v7.3'); % saving the inhibitory current  


end

