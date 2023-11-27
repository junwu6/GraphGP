import torch
import torch.nn as nn
import numpy as np
from time import process_time
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.mlls import MarginalLogLikelihood
import gpytorch
from utils.data_loader import get_data_sann


class ATGPLogLikelihood(MarginalLogLikelihood):
    def __init__(self, likelihood, model, num_source):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super(ATGPLogLikelihood, self).__init__(likelihood, model)
        self.num_source = num_source

    def _add_other_terms(self, res, params):
        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        for added_loss_term in self.model.added_loss_terms():
            res = res.add(added_loss_term.loss(*params))

        # Add log probs of priors on the (functions of) parameters
        res_ndim = res.ndim
        for name, module, prior, closure, _ in self.named_priors():
            prior_term = prior.log_prob(closure(module))
            res.add_(prior_term.view(*prior_term.shape[:res_ndim], -1).sum(dim=-1))

        return res

    def forward(self, function_dist, target, *params):
        if not isinstance(function_dist, MultivariateNormal):
            raise RuntimeError("ExactMarginalLogLikelihood can only operate on Gaussian random variables")

        # Get the log prob of the marginal distribution
        output = self.likelihood(function_dist, *params)
        mean, covar = output.mean, output.covariance_matrix
        new_mean = mean[self.num_source:]
        temp = torch.linalg.inv(covar)
        new_covar = covar[self.num_source:, :] @ temp @ covar[:, self.num_source:]
        new_output = gpytorch.distributions.MultivariateNormal(new_mean, new_covar)
        res = new_output.log_prob(target[self.num_source:])
        res = self._add_other_terms(res, params)

        # Scale by the amount of data we have
        num_data = function_dist.event_shape.numel()
        return res.div_(num_data)


class GraphGPKernel(gpytorch.kernels.Kernel):
    is_stationary = True
    has_lengthscale = True

    def __init__(self, adj=None, feat=None, num_layers=None, num_source=None, length_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(name='raw_varsigma_w', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        self.register_parameter(name='raw_sigma_w', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        self.register_parameter(name='raw_sigma_b', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))
        self.register_parameter(name='raw_s_alpha', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, len(adj), 1)))
        self.register_parameter(name='raw_t_alpha', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, len(adj), 1)))
        if length_constraint is None:
            length_constraint = Positive()
        self.register_constraint(param_name="raw_varsigma_w", constraint=length_constraint)
        self.register_constraint(param_name="raw_sigma_w", constraint=length_constraint)
        self.register_constraint(param_name="raw_sigma_b", constraint=length_constraint)
        self.register_constraint(param_name="raw_s_alpha", constraint=length_constraint)
        self.register_constraint(param_name="raw_t_alpha", constraint=length_constraint)
        self.adj = adj
        self.feat = feat
        self.num_layers = num_layers
        self.num_source = num_source

    @property
    def varsigma_w(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_varsigma_w_constraint.transform(self.raw_varsigma_w)

    @varsigma_w.setter
    def varsigma_w(self, value):
        return self._set_varsigma_w(value)

    def _set_varsigma_w(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_varsigma_w)
        self.initialize(raw_sigma_w=self.raw_varsigma_w_constraint.inverse_transform(value))

    @property
    def sigma_w(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_sigma_w_constraint.transform(self.raw_sigma_w)

    @sigma_w.setter
    def sigma_w(self, value):
        return self._set_sigma_w(value)

    def _set_sigma_w(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma_w)
        self.initialize(raw_sigma_w=self.raw_sigma_w_constraint.inverse_transform(value))

    @property
    def sigma_b(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_sigma_b_constraint.transform(self.raw_sigma_b)

    @sigma_b.setter
    def sigma_b(self, value):
        return self._set_sigma_b(value)

    def _set_sigma_b(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_sigma_b)
        self.initialize(raw_sigma_b=self.raw_sigma_b_constraint.inverse_transform(value))

    @property
    def s_alpha(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_s_alpha_constraint.transform(self.raw_s_alpha)

    @s_alpha.setter
    def s_alpha(self, value):
        return self._set_s_alpha(value)

    def _set_s_alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_s_alpha)
        self.initialize(raw_s_alpha=self.raw_s_alpha_constraint.inverse_transform(value))

    @property
    def t_alpha(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_t_alpha_constraint.transform(self.raw_t_alpha)

    @t_alpha.setter
    def t_alpha(self, value):
        return self._set_t_alpha(value)

    def _set_t_alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_t_alpha)
        self.initialize(raw_s_alpha=self.raw_t_alpha_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        x1_ = self.feat
        x2_ = self.feat
        diff = gpytorch.functions.RBFCovariance.apply(
            x1_,
            x2_,
            self.lengthscale,
            lambda x1_, x2_: self.covar_dist(x1_, x2_, square_dist=True, diag=False, **params),
        )

        K = self._gin_cov(diff)

        K_ss = K[:self.num_source, :self.num_source]
        K_tt = K[self.num_source:, self.num_source:]
        K_st = K[:self.num_source, self.num_source:]
        index_K = torch.ones_like(K)
        index_K[:self.num_source, self.num_source:] = K_st.mean()
        index_K[self.num_source:, :self.num_source] = K_st.mean()
        index_K[:self.num_source, :self.num_source] = K_ss.mean()
        index_K[self.num_source:, self.num_source:] = K_tt.mean()
        K = K.mul(index_K)

        return K[x1.flatten()][:, x2.flatten()]

    def _gin_cov(self, C0):
        K = self.varsigma_w ** 2 * self._adj_conv(C0, self.sigma_b, self.sigma_w)
        for j in range(self.num_layers - 1):
            ExxT = self._relu_cov(K)
            K = self.varsigma_w ** 2 * self._adj_conv(ExxT, self.sigma_b, self.sigma_w)
        return K

    def _adj_conv(self, K, sigma_b, sigma_w):
        K_adj = 0
        for i in range(len(self.adj)):
            for j in range(len(self.adj)):
                if i != j:  # homogeneous case
                    continue
                s_score = self.s_alpha[i]
                t_score = self.t_alpha[j]
                adj = torch.zeros_like(self.adj[0].to_dense())
                adj[:self.num_source, :self.num_source] = self.adj[i].to_dense()[:self.num_source, :self.num_source]
                adj[self.num_source:, self.num_source:] = self.adj[j].to_dense()[self.num_source:, self.num_source:]
                adj = adj.to_sparse_coo()
                temp = adj @ (adj @ K).T
                K_adj += (sigma_b ** 2 + sigma_w ** 2 * temp) * s_score * t_score
        return K_adj

    def _relu_cov(self, K):
        s = torch.sqrt(torch.diag(K)).view((-1, 1)) + 1e-5
        theta = torch.arccos(torch.clamp(K / s / s.T, -1, 1))
        ExxT = 0.5 / np.pi * (torch.sin(theta) + (np.pi - theta) * torch.cos(theta)) * s * s.T
        return ExxT


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, adj=None, feat=None, num_layers=None, num_source=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = GraphGPKernel(adj, feat, num_layers=num_layers, num_source=num_source)
        self.feat = feat

    def forward(self, idx, i):
        mean_x = self.mean_module(self.feat[idx.flatten()])
        covar_x = self.covar_module(idx)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GraphGP(nn.Module):
    def __init__(self, args, device):
        super(GraphGP, self).__init__()
        self.device = device
        self.args = args

    def fit(self, data, **params):
        X, y, A, mask = get_data_sann(data.to(self.device))
        num_source = data.s_nodes
        num_target = data.t_nodes
        now = process_time()

        best_val_acc = best_train_acc = best_test_acc = 0
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

        train_s = torch.full((num_source, 1), dtype=torch.long, fill_value=0)
        train_t = torch.full((num_target, 1), dtype=torch.long, fill_value=1)
        full_train_index = torch.cat([train_s, train_t]).to(self.device)

        ind_train = torch.where(mask["train"])[0]
        model = ExactGPModel((ind_train, full_train_index[mask["train"]]), y[mask["train"]], likelihood,
                             adj=A, feat=X, num_layers=self.args.num_layers, num_source=num_source).to(self.device)
        model.covar_module.lengthscale = torch.tensor(5.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        mll = ATGPLogLikelihood(likelihood, model, num_source=num_source)
        for epoch in range(1, 1 + self.args.epochs):
            model.train()
            likelihood.train()
            optimizer.zero_grad()
            output = model(ind_train, full_train_index[mask["train"]])

            loss = -mll(output, y[mask["train"]])
            loss.backward()
            optimizer.step()

            model.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                ind_all = torch.where(mask["train"] != torch.logical_not(mask["train"]) )[0]
                pred_output = model(ind_all, full_train_index)
                pred = likelihood(pred_output)
                result = self.evaluate(pred.mean, y, mask)
                if result["val"] > best_val_acc:
                    best_val_acc = result["val"]
                    best_test_acc = result["test"]
                    best_train_acc = result["train"]

            if epoch % 10 == 0 and self.args.verbose:
                print("Epoch: %03d, train: %.4f, val: %.4f, test: %.4f" % (epoch, result["train"], result["val"], result["test"]))

        results = torch.tensor([process_time() - now, best_train_acc, best_val_acc, best_test_acc])
        return results

    def evaluate(self, pred, y, masks):
        func = lambda fit, y: 1 - torch.sum((fit - y) ** 2) / torch.sum((y - torch.mean(y)) ** 2)
        result = {}
        for subset, mask in masks.items():
            result[subset] = func(pred[mask], y[mask])

        return result
