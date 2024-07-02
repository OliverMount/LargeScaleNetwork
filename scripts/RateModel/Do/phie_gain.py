def phie_gain(x,gain):
	import numpy as np
	g=0.16
	I=125.
	c=310.
	y=(c*x-I).*gain
	if y!=0:
		result = y./(1-np.exp(-g*y))
	else:
		result=0
	return result
