import torch 
import math

def sample_sphere(batch, radius, method='stratified', axis=1, phi_min=0, phi_max=torch.pi, theta_min=0, theta_max=2 * torch.pi):
	# assumes y-axis is up by default, otherwise we swap 
	if method == 'uniform':
		phi_range = phi_max - phi_min
		theta_range = theta_max - theta_min
		phi = torch.rand(batch, 1) * phi_range + phi_min
		theta = torch.rand(batch, 1) * theta_range + theta_min   
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
		cosPhi = torch.linspace(torch.cos(phi_min), torch.cos(phi_max)).unsqueeze(-1)
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
		cos_phi_range = math.cos(phi_min) - math.cos(phi_max)
		phi = torch.acos(math.cos(phi_min) - cos_phi_range * (i + 0.5) / batch)
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
	return sample_sphere(batch, radius, method, axis, phi_max=0.5 * torch.pi)