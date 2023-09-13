import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-l", "--lambda_loss", type=float, default=1.2,
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s", "--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")
parser.add_argument("-n", "--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r", "--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-d", "--dataset", type=str, default="cifar100",
                    help="")
parser.add_argument("-e", "--no_of_epochs", type=int, default=200,
                    help="Number of epochs for the active learner")
parser.add_argument("-m", "--method_type", type=str, default="lloss",
                    help="")
parser.add_argument("-c", "--cycles", type=int, default=5,
                    help="Number of active learning cycles")
parser.add_argument("-t", "--total", type=bool, default=False,
                    help="Training on the entire dataset")

parser.add_argument("--gcn_c_hid_dim", type=int, default=32,
                    help="GCN-C hidden dimension.")
parser.add_argument("--rsgnn_hid_dim", type=int, default=128,
                    help="RS-GNN hidden dimension.")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
parser.add_argument("--rsgnn_epochs", type=int, default=20,
                    help="Number of RS-GNN epochs.")
parser.add_argument("--gcn_c_epochs", type=int, default=1000,
                    help="Number of epochs for GCN-C.")
parser.add_argument("--num_reps_multiplier", type=int, default=100,
                    help="num_reps = num_class * num_reps_multiplier.")
parser.add_argument("--valid_each", type=int, default=1,
                    help="Validate each k epochs.")
parser.add_argument("--num_valid_nodes", type=int, default=500,
                    help="Number of validation set nodes.")
parser.add_argument("--lr", type=float, default=0.1,
                    help="Learning rate.")
parser.add_argument("--drop_rate", type=float, default=0.5,
                    help="Dropout probability.")
parser.add_argument("--gcn_c_w_decay", type=float, default=5e-4,
                    help="Weight decay for the GCN.")
parser.add_argument("--lambda_", type=float, default=0.1,
                    help="Hyperparam for JointDGI loss.")
