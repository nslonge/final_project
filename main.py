import argparse
import sys
import os
import data_utils
import model
import train
import torch
import operator
import pdb
import evaluate

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
		
# training parameters
parser = argparse.ArgumentParser(description='PyTorch project')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs for train [default: 50]')
parser.add_argument('--batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('--delta', type=float, default=.01, help='delta for use in loss function')
parser.add_argument('--save-path', type=str, default='./mod.pkl', help='where to save the snapshot')
parser.add_argument('--use-body', type=str2bool, default=False, help='use question body or just question title?')
parser.add_argument('--optimizer', type=str, default='adam', help='which optimizer to use: [default: Adam]')
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('--dataset', type=str, default='askubuntu', help='which dataset to train on [default: askubuntu]')

# model parameters
parser.add_argument('--model', type=str, default='cnn', help='use cnn or lstm model?')
parser.add_argument('--max-title', type=int, default=38, help='maximum title length [default: 38]')
parser.add_argument('--max-body', type=int, default=100, help='maximum body length [default: 100]')
parser.add_argument('--avg-pool', type=str2bool, default=False, help='use mean or max pooling [default: False]')
parser.add_argument('--embed-dim', type=int, default=200, help='number of embedding dimension [default: 200]')
parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('--kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('--neg-samples', type=int, default=20, help='number of negative samples to use in training [default: 20]')
parser.add_argument('--hidden-size', type=int, default=240, help='hidden layer size for lstm [default: 240]')
parser.add_argument('--hidden-layer', type=int, default=1, help='hidden layer number for lstm [default: 1]')
parser.add_argument('--bidirectional', type=str2bool, default=False, help='using bidirectional lstm [default: False]')
args = parser.parse_args()

def main():
	print("\nParameters:")
	for attr, value in args.__dict__.items():
		print("\t{}={}".format(attr.upper(), value))
	
	# load data
	train_data, dev_data, test_data, embeddings =\
                            data_utils.load_dataset(args, 'askubuntu-master')
	
	# initalize necessary parameters
	args.embed_num = embeddings.shape[0]
	args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
	
	# load model
	if args.snapshot is None:
            # initalize model
            if args.model == 'lstm':
				if args.bidirectional and (args.hidden_layer > 1):
					args.hidden_layer = 1
					print('\nMultilayer bidirectional LSTM not supported yet, layer set to 1.\n')
				mod = model.LSTM(args, embeddings)
            elif args.model == 'cnn':
				mod = model.CNN(args, embeddings)
            # train model
            res = train.train_model(train_data, dev_data, test_data, mod, args)

	else :
            print('\nLoading model from [%s]...' % args.snapshot)
            try:
                    mod = torch.load(args.snapshot)
            except :
                    print("Sorry, This snapshot doesn't exist."); exit()
            print(mod)
	
            # evaluate
            print('Evaluating on dev')
            evaluate.q_evaluate(mod, dev_data, args)

            print('Evaluating on test')
            evaluate.q_evaluate(mod, test_data, args)


if __name__=="__main__":
	main()


