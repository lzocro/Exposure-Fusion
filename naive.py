def naive_reconstruction(locations, w_exponents):
	'''
	Constructions the naive fusion described in section 3.2 give a sequence of images and the exponenets [w_c,w_s,w_e]
	returns a single fused image

	We have a problem when all the weights for all three pictures become 0. What to do? average all three pictures, keep only missing pixels and add to the picture? 

	'''
	assert len(w_exponents) == 3, 'Incorrect dimension of w_exponents'
	image = read_sequence_to_fuse(locations)
	W = (constrast_measure(image)**w_exponents[0])*(saturation_measure(image)**w_exponents[1])*(exposure_measure(image)**w_exponents[2])
	a,b,c = W.shape
	W_prod = np.empty((a,b,c,3), dtype='float')
	for i in range(W.shape[0]):
		W[i] = np.multiply( 1/(np.sum(W, axis = 0)), W[i]) #wtilda
	for i in range(W.shape[0]):	
		W_prod[i] = np.stack((W[i],W[i],W[i]), axis = -1)
	R = np.uint8(np.sum(np.multiply(W_prod, image), axis = 0))
	return(R)
