Spath='/olive/Maths/R/Do/SNN/Results/Spikes/'

for roi in range(N_ROI):
	for neu in range(N_e):
		spk_train= Spk[roi][Spk[roi][:,1]==neu+(roi*N_e),0].astype('uint16')
		spk_train=spk_train[spk_train<(sig_len-1)]
		SPK_TRAINS[:,neu,roi]=SpkTime2Binary(spk_train,sig_len)
	print("---> Storing 0 and 1 spikes of ROI-"+str(roi))

print("---> Saving the spike time series...")
np.save(Spath+'SPK_TRAIN_Participant'+str(participant_number)+'.npy',SPK_TRAINS)


#with open(Spath+'SPK_TRAIN_Participant'+str(participant_number)+'.npy', 'rb') as f: #		SPK_TRAINS=np.load(f)
