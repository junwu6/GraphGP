from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import add_self_loops
import torch
from torch_sparse import coalesce
from torch_scatter import scatter_add
import numpy as np
from torch_geometric.utils import add_remaining_self_loops, scatter


class transition_matrix(BaseTransform):
    """
    Returns a transform that applies normalization on a given sparse matrix.
    Args:
        self_loop_weight (float): weight of the added self-loop. (default: `1`)
        normalization (str): Normalization scheme. supported values:
            "sym": symmetric normalization $\hat{A}=D^{-1/2}AD^{-1/2}$.
            "col": column-wise normalization $\hat{A}=AD^{-1}$.
            "row": row-wise normalization $\hat{A}=D^{-1}A$.
            others: No normalization.
            (default: "sym")
    """

    def __init__(self, self_loop_weight=1, normalization="sym"):
        self.self_loop_weight = self_loop_weight
        self.normalization = normalization

    def __call__(self, data):
        N = data.num_nodes
        A = data.edge_index
        if data.edge_attr is None:
            edge_weight = torch.ones(A.size(1), device=A.device)
        else:
            edge_weight = data.edge_attr

        # if self.self_loop_weight:
        #     A, edge_weight = add_self_loops(A, edge_weight, fill_value=self.self_loop_weight, num_nodes=N)

        # A, edge_weight = coalesce(A, edge_weight, N, N)
        if self.normalization == "sym":
            if self.self_loop_weight:
                A, edge_weight = add_self_loops(A, edge_weight, fill_value=self.self_loop_weight, num_nodes=N)
            row, col = A
            deg = scatter_add(edge_weight, col, dim=0, dim_size=N)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        elif self.normalization == "col":
            if self.self_loop_weight:
                A, edge_weight = add_self_loops(A, edge_weight, fill_value=self.self_loop_weight, num_nodes=N)
            _, col = A
            deg = scatter_add(edge_weight, col, dim=0, dim_size=N)
            deg_inv = 1. / deg
            deg_inv[deg_inv == float("inf")] = 0
            edge_weight = edge_weight * deg_inv[col]
        elif self.normalization == "row":
            if self.self_loop_weight:
                A, edge_weight = add_self_loops(A, edge_weight, fill_value=self.self_loop_weight, num_nodes=N)
            row, _ = A
            deg = scatter_add(edge_weight, row, dim=0, dim_size=N)
            deg_inv = 1. / deg
            deg_inv[deg_inv == float("inf")] = 0
            edge_weight = edge_weight * deg_inv[row]
        else:
            pass

        data.edge_index = A
        data.edge_attr = edge_weight

        return data


# class transition_matrix(BaseTransform):
#     """
#     Returns a transform that applies normalization on a given sparse matrix.
#     Args:
#         self_loop_weight (float): weight of the added self-loop. (default: `1`)
#         normalization (str): Normalization scheme. supported values:
#             "sym": symmetric normalization $\hat{A}=D^{-1/2}AD^{-1/2}$.
#             "col": column-wise normalization $\hat{A}=AD^{-1}$.
#             "row": row-wise normalization $\hat{A}=D^{-1}A$.
#             others: No normalization.
#             (default: "sym")
#     """
#
#     def __init__(self, self_loop_weight=1, normalization="sym"):
#         self.self_loop_weight = self_loop_weight
#         self.normalization = normalization
#
#     def __call__(self, data):
#         # N = data.num_nodes
#         # A = data.edge_index
#         # if data.edge_attr is None:
#         #     edge_weight = torch.ones(A.size(1), device=A.device)
#         # else:
#         #     edge_weight = data.edge_attr
#         #
#         # if self.self_loop_weight:
#         #     A, edge_weight = add_self_loops(A, edge_weight, fill_value=self.self_loop_weight, num_nodes=N)
#
#         # A, edge_weight = coalesce(A, edge_weight, N, N)
#         if self.normalization == "sym":
#             # row, col = A
#             # deg = scatter_add(edge_weight, col, dim=0, dim_size=N)
#             # deg_inv_sqrt = deg.pow(-0.5)
#             # deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
#             # edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
#
#             edge_index = data.edge_index
#             num_nodes = data.x.shape[0]
#             edge_weight = None
#             edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1., num_nodes)
#             edge_weight = torch.ones((edge_index.size(1), ), dtype=data.x.dtype, device=edge_index.device)
#             row, col = edge_index[0], edge_index[1]
#             deg = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce='sum')
#             deg_inv_sqrt = deg.pow_(-0.5)
#             deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
#             edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
#         # elif self.normalization == "col":
#         #     _, col = A
#         #     deg = scatter_add(edge_weight, col, dim=0, dim_size=N)
#         #     deg_inv = 1. / deg
#         #     deg_inv[deg_inv == float("inf")] = 0
#         #     edge_weight = edge_weight * deg_inv[col]
#         elif self.normalization == "row":
#             edge_index = data.edge_index
#             num_nodes = data.x.shape[0]
#             edge_weight = None
#             edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1., num_nodes)
#             edge_weight = torch.ones((edge_index.size(1), ), dtype=data.x.dtype, device=edge_index.device)
#             row, col = edge_index[0], edge_index[1]
#             deg = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce='sum')
#             deg_inv = 1. / deg
#             deg_inv[deg_inv == float("inf")] = 0
#             edge_weight = edge_weight * deg_inv[row]
#             # row, _ = A
#             # deg = scatter_add(edge_weight, row, dim=0, dim_size=N)
#             # deg_inv = 1. / deg
#             # deg_inv[deg_inv == float("inf")] = 0
#             # edge_weight = edge_weight * deg_inv[row]
#         else:
#             edge_index = data.edge_index
#             if data.edge_attr is None:
#                 edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
#             else:
#                 edge_weight = data.edge_attr
#
#         data.edge_index = edge_index
#         data.edge_attr = edge_weight
#         return data


def set_random_seed(seed=0):
    # seed setting
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def result_format(result):
    return "time: %.4f, train: %.4f, val: %.4f, test: %.4f" % (result[0], result[1], result[2], result[3])
