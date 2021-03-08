import argparse
import os
from solver import Solver
import constant
import environment

# ../output/hiv/model_951.pth
def run():
	parser = argparse.ArgumentParser()
	'''Model configuration'''
	parser.add_argument("--input_dim", type=int, default=512)
	parser.add_argument("--hidden_dim", type=int, default=128)
	parser.add_argument("--out_dim", type=int, default=128)
	parser.add_argument("--layer_hidden", type=int, default=2)
	parser.add_argument("--layer_output", type=int, default=3)
	parser.add_argument("--skip_connection_type", type=str, default="gsc")
	parser.add_argument("--norm_type", type=str, default="no")
	
	'''Training configuration'''
	parser.add_argument("--batch_train", type=int, default=128)
	parser.add_argument("--batch_test", type=int, default=128)
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--lr_decay", type=float, default=0.99)
	parser.add_argument("--decay_interval", type=int, default=10)
	parser.add_argument("--iteration", type=int, default=100)
	parser.add_argument("--loss_type", type=str, default="FocalLoss")
	parser.add_argument("--load_model_path", help='Write the address if you want to load the model.', type=str, default="")

	'''Miscellaneous'''
	parser.add_argument("--mode", type=str, help='Choose either train or test', default="train")
	parser.add_argument("--dataset", type=str, help='Dataset to use', default="bionsight")
	parser.add_argument("--model_type", type=str, help='Used to create a folder name', default="GNN")
	parser.add_argument("--use_augmentation", help='Augmentation using Smiles Enumeration', type=bool, default=True)
	parser.add_argument("--radius", type=int, help='The number of times each node ID is updated', default=1)
	args = parser.parse_args()

	environment.setup(args)	
	'''Solver for training and testing MolGNN'''
	solver = Solver()
	solver.run()


if __name__ == "__main__":
	run()