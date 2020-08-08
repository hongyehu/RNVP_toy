import numpy as np
import matplotlib.pyplot as plt
import torch

def generate_laplace(batch_size=200):
	scale = torch.tensor(1/np.sqrt(2),dtype = torch.float32)
	shape = [batch_size]+[2]
	finfo = torch.finfo(torch.float32)
	u = scale.new_empty(shape).uniform_(finfo.eps - 1, 1)
	out = scale * u.sign() * torch.log1p(-u.abs())
	return out

def generate_pinwheel(batch_size=200):
	radial_std = 0.4
	tangential_std = 0.1
	num_classes = 4
	num_per_class = batch_size // 4
	rate = 0.25
	rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

	features = np.random.randn(num_classes*num_per_class, 2) \
		* np.array([radial_std, tangential_std])
	features[:, 0] += 1.
	labels = np.repeat(np.arange(num_classes), num_per_class)

	angles = rads[labels] + rate * np.exp(features[:, 0])
	rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
	rotations = np.reshape(rotations.T, (-1, 2, 2))

	return 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))


# if __name__ == '__main__':

	# data = generate_pinwheel(1000)
	# data = generate_laplace(10000)
	# data = data.numpy()
	# plt.figure()
	# plt.scatter(data[:,0],data[:,1],alpha=0.5)
	# plt.savefig('./laplace.jpg')