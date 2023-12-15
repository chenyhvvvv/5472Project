import torch
import torch.nn as nn

class PoissonLoss(nn.Module):
    def __init__(self, node_features, basis, count_matrix, library_size, coef_fe):
        super(PoissonLoss, self).__init__()
        self.node_features = node_features
        self.basis = basis
        self.count_matrix = count_matrix
        self.library_size = library_size
        self.coef_fe = coef_fe

    def forward(self, recon_node_features, beta, alpha, gamma):
        feat_recon_loss = torch.mean(torch.sqrt(torch.sum(torch.pow(recon_node_features - self.node_features, 2), axis=1)))
        log_lam = torch.log(torch.matmul(beta, self.basis) + 1e-6) + alpha + gamma
        lam = torch.exp(log_lam)
        decon_loss = - torch.mean(torch.sum(self.count_matrix *
                                           (torch.log(self.library_size + 1e-6) + log_lam) - self.library_size * lam, axis=1))
        return decon_loss + self.coef_fe * feat_recon_loss