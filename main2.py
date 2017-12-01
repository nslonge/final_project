import argparse
import sys
import os
import data_utils
import model
import train2
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
parser.add_argument('--batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('--shuffle', action='store_true', default=False, help='shuffle the data every epoch' )
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for train [default: 10]')
parser.add_argument('--save-path', type=str, default='./mod.pkl', help='where to save the snapshot')
parser.add_argument('--snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('--optimizer', type=str, default='adam', help='which optimizer to use: [default Adam]')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate [default: 0.001]')

# model parameters
parser.add_argument('--model', type=str, default='cnn', help='use cnn or lstm model?')
parser.add_argument('--max-title', type=int, default=20, help='maximum title length [default: 38]')
parser.add_argument('--avg-pool', type=str2bool, default=True, help='use mean or max pooling [default: False]')
parser.add_argument('--delta', type=float, default=.01, help='delta for use in loss function')
parser.add_argument('--lambd', type=float, default=.1, help='lambda value form use in gradient reversal')
parser.add_argument('--use-body', type=str2bool, default=False, help='use question body or just question title?')
parser.add_argument('--max-body', type=int, default=100, help='maximum body length [default: 100]')
parser.add_argument('--embed-dim', type=int, default=300, help='number of embedding dimension [default: 200]')
parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('--kernel-sizes', type=str, default='2,3,4', help='comma-separated kernel size to use for convolution')
parser.add_argument('--hidden-size', type=int, default=100, help='number of hidden layer size')
parser.add_argument('--domain-size', type=int, default=100, help='hidden layer size in domain classifier')
parser.add_argument('--neg-samples', type=int, default=20, help='number of negative samples to use in training')
args = parser.parse_args()

def main():
        #args.train_sim = False
	print("\nParameters:")
	for attr, value in args.__dict__.items():
		print("\t{}={}".format(attr.upper(), value))
	
	# load data
	strain_data, sdev_data, stest_data, embeddings =\
                            data_utils.load_dataset(args, 'askubuntu-master')
	dtrain_data, ddev_data, dtest_data, _ =\
                            data_utils.load_dataset(args, 'Android-master')

	# initalize necessary parameters
	args.embed_num = embeddings.shape[0]
	args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
	
	# load model
	if args.snapshot is None:
            # initalize model
            task_model = None
            if args.model == 'lstm':
                    task_model = model.LSTM(args, embeddings)
            elif args.model == 'cnn':
                    task_model = model.CNN(args, embeddings)

            domain_model = model.DomainClassifier(args, embeddings)

            # train models
            res = train2.train_model(strain_data, sdev_data, stest_data, 
                                     dtrain_data, ddev_data, dtest_data,
                                     task_model, domain_model, args)
	else :
            print('\nLoading model from [%s]...' % args.snapshot)
            try:
                    mod = torch.load(args.snapshot)
            except :
                    print("Sorry, This snapshot doesn't exist."); exit()
            print(mod)
	
            # evaluate
            print('Evaluating on dev')
            evaluate.evaluate(mod, dev_data, args)

            print('Evaluating on test')
            evaluate.evaluate(mod, test_data, args)

if __name__=="__main__":
	main()


