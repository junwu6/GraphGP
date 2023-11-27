import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import numpy as np
import copy
from torch_geometric.datasets import Twitch
from torch_geometric.datasets import Airports
from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.datasets import WebKB


def degree_bucketing(graph, max_degree=256):
    features = torch.zeros([graph.number_of_nodes(), max_degree])
    for i in range(graph.number_of_nodes()):
        try:
            features[i][min(graph.in_degree(i), max_degree-1)] = 1
        except:
            features[i][0] = 1
    return features


def load_data(data_name, domain_name, path='../datasets', task=None, transform=None):
    dataset, data = None, None
    if data_name == "Airports":
        dataset = Airports(path, name=domain_name)
        data = dataset[0]
    elif data_name == "Twitch":
        dataset = Twitch(path, name=domain_name)
        data = dataset[0]
        # using regression output
        with open(f"{path}/{domain_name}/{domain_name}_target.csv", "r") as f:
            t = f.read().split("\n")[1:-1]
            yll = {int(r.split(",")[5]): int(r.split(",")[3]) for r in t}
            num_views = []
            for i in range(data.x.shape[0]):
                if i in yll:
                    num_views.append(yll[i])
                else:
                    num_views.append(0)
            data.y = torch.log(torch.tensor(num_views)+1)
    elif data_name == "WikipediaNetwork":
        dataset = WikipediaNetwork(path, name=domain_name, geom_gcn_preprocess=False)
        data = dataset[0]
    elif data_name == "WebKB":
        dataset = WebKB(path, name=domain_name)
        data = dataset[0]
    else:
        print("Unknown data!")

    if task == "classification":
        data.num_classes = dataset.num_classes

    if data_name == "Airports":
        G = to_networkx(data)
        data.x = degree_bucketing(G)

    if transform is not None:
        data = transform(data)

    return data


def get_data(data):
    """
    Return attributes from a homogeneous graph.
    Args:
        data (torch_geometric.data.data): a graph data object.
    """
    X = data.x
    y = data.y
    N = X.shape[0]
    if data.edge_attr is None:
        edge_weight = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
    else:
        edge_weight = data.edge_attr
    A = torch.sparse_coo_tensor(data.edge_index, edge_weight, (N, N))
    mask = {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}
    return X, y, A, mask


def get_data_sann(data):
    """
    Return attributes from a homogeneous graph.
    Args:
        data (torch_geometric.data.data): a graph data object.
    """
    X = data.x
    y = data.y
    N = X.shape[0]
    from torch_geometric.utils import remove_self_loops
    edge_weight = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
    edge_index, edge_weight = remove_self_loops(data.edge_index, edge_weight)

    A = torch.sparse_coo_tensor(edge_index, edge_weight, (N, N))

    adjs = [torch.eye(A.shape[0]).to_sparse().to(A.device)]
    for k in range(2):
        A = A + adjs[0]
        rowsum = A.to_dense().sum(1)
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        r_mat_inv = r_mat_inv.to_sparse()
        norm_A = torch.sparse.mm(r_mat_inv, A)
        adjs.append(norm_A)

        B = torch.mm(A.to_dense(), A.to_dense())
        B = B.to_sparse()
        B = torch.sparse_coo_tensor(B.indices(), torch.ones_like(B.values()), (N, N))
        A = B - B.mul(A)

    mask = {"train": data.train_mask, "val": data.val_mask, "test": data.test_mask}
    return X, y, adjs, mask


def preprocessing(source_graph, target_graph, transform=None, ratio=0.01, task=None, mode="mix", seed=0):
    num_source_nodes = source_graph.x.shape[0]
    x = torch.cat([source_graph.x, target_graph.x], dim=0)
    y = torch.cat([source_graph.y, target_graph.y], dim=0)
    if task == "regression":
        y = y.float()
    edge_index = torch.cat([source_graph.edge_index, target_graph.edge_index+num_source_nodes], dim=1)
    data = Data(x=x, edge_index=edge_index, y=y)
    data.s_nodes = source_graph.x.shape[0]
    data.t_nodes = target_graph.x.shape[0]
    if task == "classification":
        data.num_classes = source_graph.num_classes

    np.random.seed(seed)
    print("Number of Source: {}, Number of Labeled Target: {}".format(num_source_nodes, int(ratio*target_graph.x.shape[0])))
    source_mask = torch.ones(source_graph.x.shape[0], dtype=torch.bool)
    target_mask = torch.zeros(target_graph.x.shape[0], dtype=torch.bool)
    ridx = np.arange(target_graph.x.shape[0])
    np.random.shuffle(ridx)

    train_mask = copy.deepcopy(target_mask)
    if mode == "target_only":
        train_mask[ridx[:int(ratio*target_graph.x.shape[0])]] = True
        data.train_mask = torch.cat([torch.logical_not(source_mask), train_mask], dim=0)
    elif mode == "source_only":
        data.train_mask = torch.cat([source_mask, train_mask], dim=0)
    else:
        train_mask[ridx[:int(ratio*target_graph.x.shape[0])]] = True
        data.train_mask = torch.cat([source_mask, train_mask], dim=0)

    val_mask = copy.deepcopy(target_mask)
    val_mask[ridx[int(ratio * target_graph.x.shape[0]): 2 * int(ratio * target_graph.x.shape[0])]] = True
    data.val_mask = torch.cat([torch.logical_not(source_mask), val_mask], dim=0)

    test_mask = copy.deepcopy(target_mask)
    test_mask[ridx[2 * int(ratio * target_graph.x.shape[0]):]] = True
    data.test_mask = torch.cat([torch.logical_not(source_mask), test_mask], dim=0)

    if transform is not None:
        data = transform(data)
    return data
