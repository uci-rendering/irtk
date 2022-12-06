import torch 

def sample_sphere(batch, radius, method='stratified', axis=1):
	# assumes y-axis is up by default, otherwise we swap 
	if method == 'uniform':
		phi = torch.rand(batch, 1) * torch.pi 
		theta = torch.rand(batch, 1) * 2 * torch.pi     
		sinPhi = torch.sin(phi)  
		cosPhi = torch.cos(phi) 
		sinTheta = torch.sin(theta) 
		cosTheta = torch.cos(theta) 
		samples = torch.cat([
			sinPhi * cosTheta, 
			cosPhi,
			sinPhi * sinTheta
		], dim=1) * radius

	elif method == 'stratified':
		cosPhi = torch.linspace(1.0 - 0.01, -1.0 + 0.01, batch).unsqueeze(-1)
		theta  = torch.linspace(0, torch.pi * 10, batch).unsqueeze(-1) \
			+ torch.rand(batch, 1) * 0.01
		
		sinPhi = torch.sqrt(1 - cosPhi * cosPhi)
		sinTheta = torch.sin(theta) 
		cosTheta = torch.cos(theta) 
		samples = torch.cat([
			sinPhi * cosTheta, 
			cosPhi,
			sinPhi * sinTheta
		], dim=1) * radius
		
	elif method == 'fibonacci':
		# From http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
		golden_ratio = (1 + 5**0.5) / 2
		i = torch.arange(batch).unsqueeze(-1)
		theta = 2 * torch.pi * i / golden_ratio
		phi = torch.acos(1 - 2 * (i + 0.5) / batch)
		sinPhi = torch.sin(phi)  
		cosPhi = torch.cos(phi)
		sinTheta = torch.sin(theta) 
		cosTheta = torch.cos(theta) 
		samples = torch.cat([
			sinPhi * cosTheta, 
			cosPhi,
			sinPhi * sinTheta
		], dim=1) * radius
	
	# Switch axis 
	index = torch.LongTensor([0, 1, 2])
	index[axis] = 1 
	index[1] = axis 
	points = samples[:, index]

	return points

def sample_hemisphere(batch, radius, method='stratified', axis=1):
	# assumes y-axis is up by default, otherwise we swap 
	if method == 'uniform':
		phi = torch.rand(batch, 1) * torch.pi * 0.5
		theta = torch.rand(batch, 1) * 2 * torch.pi     
		sinPhi = torch.sin(phi)  
		cosPhi = torch.cos(phi) 
		sinTheta = torch.sin(theta) 
		cosTheta = torch.cos(theta) 
		samples = torch.cat([
			sinPhi * cosTheta, 
			cosPhi,
			sinPhi * sinTheta
		], dim=1) * radius

	elif method == 'stratified':
		cosPhi = torch.linspace(1.0 - 0.01, 0.0 + 0.01, batch).unsqueeze(-1)
		theta  = torch.linspace(0, torch.pi * 10, batch).unsqueeze(-1) \
			+ torch.rand(batch, 1) * 0.01
		
		sinPhi = torch.sqrt(1 - cosPhi * cosPhi)
		sinTheta = torch.sin(theta) 
		cosTheta = torch.cos(theta) 
		samples = torch.cat([
			sinPhi * cosTheta, 
			cosPhi,
			sinPhi * sinTheta
		], dim=1) * radius
		
	elif method == 'fibonacci':
		# From http://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
		golden_ratio = (1 + 5**0.5) / 2
		i = torch.arange(batch).unsqueeze(-1)
		theta = 2 * torch.pi * i / golden_ratio
		phi = torch.acos(1 - (i + 0.5) / batch)
		sinPhi = torch.sin(phi)  
		cosPhi = torch.cos(phi)
		sinTheta = torch.sin(theta) 
		cosTheta = torch.cos(theta) 
		samples = torch.cat([
			sinPhi * cosTheta, 
			cosPhi,
			sinPhi * sinTheta
		], dim=1) * radius
	
	# Switch axis 
	index = torch.LongTensor([0, 1, 2])
	index[axis] = 1 
	index[1] = axis 
	points = samples[:, index]

	return points