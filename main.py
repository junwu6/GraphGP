from argparse import ArgumentParser
import torch
from utils.helper import transition_matrix, result_format
from utils.data_loader import load_data, preprocessing
from GP.GraphGP import GraphGP


parser = ArgumentParser(description="Transferable Graph Learning")
parser.add_argument("--data", type=str, default='Twitch')
parser.add_argument("--source", type=str, default='RU')
parser.add_argument("--target", type=str, default='PT')
parser.add_argument("--method", type=str, default='GraphGP', help='model name')
parser.add_argument("--task", type=str, default='regression')
parser.add_argument("--verbose", type=bool, default=True, help='print results')
parser.add_argument('-cuda', type=int, default=0, help='cuda id')
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=500)
args = parser.parse_args()


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
    print("Running {} on {} Data Set!".format(args.method, args.data))

    # read source and target graphs
    transform = transition_matrix(normalization="row")
    source_graph = load_data(args.data, args.source, task=args.task).to(device)
    target_graph = load_data(args.data, args.target, task=args.task).to(device)
    data = preprocessing(source_graph, target_graph, task=args.task, transform=transform)

    # run the model
    instance_model = globals()[args.method]
    model = instance_model(args, device)
    result_runs = model.fit(data)
    print("----")
    print("Final Result:   ", result_format(result_runs))
    print("----")
