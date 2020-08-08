import os
import time
import traceback
from math import log, sqrt

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.utils import clip_grad_norm_

import layer
import source
import utils
from args import args
from utils import my_log
from toy_data import generate_pinwheel

def RNVP_init():
	if args.prior == 'gaussian':
		prior = source.Gaussian([args.nvars])
	elif args.prior == 'cauchy':
		prior = source.Cauchy([args.nvars])
	elif args.prior == 'laplace':
		prior = source.Laplace([args.nvars])
	else:
		raise ValueError('Unknown prior: {}'.format(args.prior))
	widths = ([args.nvars] + [args.nhidden] * args.nmlp +
				  [args.nvars])
	flow = layer.RNVP(args.nvars,
				[
					layer.ResNet(args.nresblocks,
										widths,
										final_scale=True,
										final_tanh=True)
					for _ in range(args.nlayers)
				],
				[
					layer.ResNet(args.nresblocks,
										widths,
										final_scale=True,
										final_tanh=False)
					for _ in range(args.nlayers)
				],
				prior

		)
	flow.to(args.device)
	return flow
def main():
	start_time = time.time()

	utils.init_out_dir()
	last_epoch = utils.get_last_checkpoint_step()
	if last_epoch >= args.epoch:
		exit()
	if last_epoch >= 0:
		my_log('\nCheckpoint found: {}\n'.format(last_epoch))
	else:
		utils.clear_log()
	utils.print_args()
	flow = RNVP_init()
	flow.train(True)
	my_log('Total nparams: {}'.format(utils.get_nparams(flow)))
	# Use multiple GPUs
	if args.cuda and torch.cuda.device_count() > 1:
		flow = utils.data_parallel_wrap(flow)

	params = [x for x in flow.parameters() if x.requires_grad]
	optimizer = torch.optim.AdamW(params,
								  lr=args.lr,
								  weight_decay=args.weight_decay)

	if last_epoch >= 0:
		utils.load_checkpoint(last_epoch, flow, optimizer)
	init_time = time.time() - start_time
	my_log('init_time = {:.3f}'.format(init_time))

	my_log('Training...')
	start_time = time.time()
	for epoch in range(last_epoch + 1, args.epoch + 1):
		optimizer.zero_grad()
		samples = generate_pinwheel(args.batch_size)
		samples = torch.tensor(samples,dtype = torch.float32).to(args.device)
		log_prob = flow.log_prob(samples)
		loss = -log_prob
		loss_mean = loss.mean()
		print('loss:', loss_mean.item())
		loss_mean.backward()
		if args.clip_grad:
				clip_grad_norm_(params, args.clip_grad)
		optimizer.step()
		if (args.out_filename and args.save_epoch
				and epoch % args.save_epoch == 0):
			state = {
				'flow': flow.state_dict(),
				'optimizer': optimizer.state_dict(),
			}
			torch.save(state,
					   '{}_save/{}.state'.format(args.out_filename, epoch))
			if epoch > 0 and (epoch - 1) % args.keep_epoch != 0:
				os.remove('{}_save/{}.state'.format(args.out_filename,
													epoch - 1))
		if (args.plot_filename and args.plot_epoch
				and epoch % args.plot_epoch == 0):
			with torch.no_grad():
				test_sample,_ = flow.sample(1000)
				test_sample = test_sample.detach().cpu().numpy()
				plt.figure()
				plt.scatter(test_sample[:,0],test_sample[:,1],alpha=0.5)
				plt.savefig(args.plot_filename+'/{}.jpg'.format(epoch))
				plt.close()






if __name__ == '__main__':
	main()


	














