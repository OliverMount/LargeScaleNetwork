clear all;clc; close all;
addpath('/olive/Maths/R/Do/')  
load /olive/Maths/R/Do/InputData/DKatlas_timeseries.mat
load /olive/Maths/R/Do/InputData/DKcortex_selectedGenes.mat;

load /olive/Maths/R/Do/InputData/SC_GenCog_PROB_30.mat
load /olive/Maths/R/Do/InputData/FCtdata2.mat 
load /olive/Maths/R/Do/InputData/FCdata.mat
 
 
fprintf('++ Loaded Gene, time series and SC data \n');

% Processing the genedata

coefe=sum(expMeasures(:,18:25),2);%./sum(expMeasures(:,[2:9 12:14]),2); %% 18:21 ampa+ 22:25 nmda/gaba
coefe;
ratioE=coefe/(max(coefe));
ratioE(35:68)=ratioE(1:34);

coefi=sum(expMeasures(:,[2:9 12:14]),2); %% 18:21 ampa+ 22:25 nmda/gaba
ratioI=coefi/(max(coefi));
ratioI(35:68)=ratioI(1:34);
ratio=ratioE./ratioI;
ratio=ratio/(max(ratio)-min(ratio));
ratio=ratio-max(ratio)+1;

%load /olive/Maths/R/Do/InputData/APARC_genedata.mat
%coefei=data;
%ratio=coefei/(max(coefei)-min(coefei));
%ratio=ratio-max(ratio)+1;
%ratio(35:68)=ratio(1:34);


disp("++ Processed the gene data")

C=GrCV([1:34 42:75],[1:34 42:75]);
C=C/max(max(C))*0.2;

N=68;
NSUB=389;
NSUBSIM=389; 
Tmax=616;
indexsub=1:NSUB; 

FC_emp=squeeze(mean(FCdata,1));

FCemp2=FC_emp-FC_emp.*eye(N);
GBCemp=mean(FCemp2,2);

Isubdiag = find(tril(ones(N),-1));

%%%%%%%%%%%%%%

flp = .008;           % lowpass frequency of filter
fhi = .08;           % highpass
delt = 0.754;            % sampling interval
k=2;                  % 2nd order butterworth filter
fnq=1/(2*delt);       % Nyquist frequency
Wn=[flp/fnq fhi/fnq]; % butterworth bandpass non-dimensional frequency
[bfilt2,afilt2]=butter(k,Wn);   % construct the filter
 

%%%% 
 
 
FCtdata=squeeze(nanmean(FCtdata2,1));
 
% intialization for storing simulated values 
FCt2=zeros(NSUBSIM,N,N);
stausi=zeros(NSUBSIM,N);

