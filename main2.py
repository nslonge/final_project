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
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate [default: 0.01]')
parser.add_argument('--lr-d', type=float, default=0.01, help='initial learning rate for domain classifier [default: 0.01]')
parser.add_argument('--full-eval', type=str2bool, default=True, help='run full adversarial domain adaptation')

# model parameters
parser.add_argument('--model', type=str, default='cnn', help='use cnn or lstm model? [default: cnn]')
parser.add_argument('--max-title', type=int, default=20, help='maximum title length [default: 20]')
parser.add_argument('--avg-pool', type=str2bool, default=False, help='use mean or max pooling [default: True]')
parser.add_argument('--delta', type=float, default=.01, help='delta for use in loss function [default: 0.01]')
parser.add_argument('--lambd', type=float, default=.01, help='lambda value form use in gradient reversal [default: 0.01]')
parser.add_argument('--use-body', type=str2bool, default=False, help='use question body or just question title?')
parser.add_argument('--max-body', type=int, default=100, help='maximum body length [default: 100]')
parser.add_argument('--embed-dim', type=int, default=300, help='number of embedding dimension [default: 300]')
parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel [default: 100]')
parser.add_argument('--kernel-sizes', type=str, default='2,3,4', help='comma-separated kernel size to use for convolution')
parser.add_argument('--hidden-size', type=int, default=240, help='hidden layer size [default: 240]')
parser.add_argument('--hidden-layer', type=int, default=1, help='hidden layer number for lstm [default: 1]')
parser.add_argument('--bidirectional', type=str2bool, default=False, help='using bidirectional lstm [default: False]')
parser.add_argument('--domain-size', type=int, default=100, help='hidden layer size in domain classifier [default: 100]')
parser.add_argument('--neg-samples', type=int, default=20, help='number of negative samples to use in training [default; 20]')
parser.add_argument('--decay-lr', type=str2bool, default=False, help='decay learning rate over time')
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
				if args.bidirectional and (args.hidden_layer > 1):
					args.hidden_layer = 1
					print('\nMultilayer bidirectional LSTM not supported yet, layer set to 1.\n')
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
                        
            print('\nEvaluating on source dev')
            evaluate.q_evaluate(mod, sdev_data, args)
            
            print('Evaluating on source test')
            evaluate.q_evaluate(mod, stest_data, args)
            
            print('\nEvaluating on target dev')
            evaluate.q_evaluate(mod, ddev_data, args)
            
            print('Evaluating on target test')
            evaluate.q_evaluate(mod, dtest_data, args)
            
if __name__=="__main__":
	main()


