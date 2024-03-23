import argparse
import yaml, os

CONFIG_PATH = "cfg/"

# Function to load yaml configuration file
def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


arg_lists = []


def get_config(argv=None,ensemble=False,dropout=False,variational=False,agentformer=False):

	# Parser arguments
	parser = argparse.ArgumentParser(description='')

	def add_argument_group(name):
		arg = parser.add_argument_group(name)
		arg_lists.append(arg)
		return arg

	# Training / test parameters
	training_args = add_argument_group('Training')
	training_args.add_argument('--num-ensembles',
							type=int, default=5, metavar='N',
							help='number of elements in the ensemble (default: 5)')
	if dropout:
		training_args.add_argument('--dropout-rate',
									type=int, default=0.5, metavar='N',
									help='dropout rate (default: 0.5)')
		training_args.add_argument('--dropout-samples',
									type=int, default=100, metavar='N',
									help='number of elements in the ensemble (default: 100)')
	if variational:
		training_args.add_argument('--num-mctrain',
					type=int, default=5, metavar='N',
					help='number of sample monte carlo for train (default: 5)')
		training_args.add_argument('--num-mctest',
					type=int, default=5, metavar='N',
					help='number of monte carlo for test (default: 5)')
	training_args.add_argument('--batch-size', '--b',
						type=int, default=256, metavar='N',
						help='input batch size for training (default: 256)')
	if not agentformer:
		training_args.add_argument('--epochs', '--e',type=int, default=100, metavar='N',
								help='number of epochs to train (default: 200)')
	else:
		training_args.add_argument('--epochs', default=None)

	training_args.add_argument('--learning-rate', '--lr',
						type=float, default=0.0004, metavar='N',
						help='learning rate of optimizer (default: 1E-3)')
	training_args.add_argument('--teacher-forcing',
						action='store_true',
						help='uses teacher forcing during training')
	training_args.add_argument('--no-retrain',
						action='store_true',
						help='do not retrain the model')
	training_args.add_argument('--validation-proportion', '--vp',
						type=float, default=0.1, metavar='N',
						help='validation proportion out of training set (default: 0.1)')

	# Data arguments
	data_args = add_argument_group('Data')
	data_args.add_argument('--id-dataset',
						type=str, default=0, metavar='N',
						help='id of the dataset to use. 0 is ETH-UCY, 1 is SDD (default: 0)')
	data_args.add_argument('--id-test',
						type=int, default=2, metavar='N',
						help='id of the dataset to use as test in LOO (default: 2)')
	data_args.add_argument('--pickle',
						action='store_true',
						help='use previously made pickle files')
	data_args.add_argument('--max-overlap',
						type=int, default=1, metavar='N',
						help='Maximal overlap between trajets (default: 1)')
	data_args.add_argument('--no_neighbors',
						action='store_true',
						help='Do not compute the neighbors')

	# Visualization arguments
	visualization_args = add_argument_group('Visualization')
	visualization_args.add_argument('--examples',
						type=int, default=0, metavar='N',
						help='number of examples to exhibit (default: 0)')
	visualization_args.add_argument('--show-plot', default=False,
						action='store_true', help='show the test plots')
	visualization_args.add_argument('--plot-losses',
						action='store_true',
						help='plot losses curves after training')
	if agentformer:
		agentformer_args = add_argument_group('Agentformer')
		agentformer_args.add_argument('--data_eval', default='test')
		agentformer_args.add_argument('--cached', action='store_true', default=False)
		agentformer_args.add_argument('--cleanup', action='store_true', default=False)

	misc_args = add_argument_group('Misc')
	misc_args.add_argument('--seed',type=int, default=1,help='Random seed for all randomized functions')
	misc_args.add_argument('--log-level',type=int, default=20,help='Log level (default: 20)')
	misc_args.add_argument('--log-file',default='',help='Log file (default: standard output)')
	
	args = parser.parse_args(argv)
	return args
