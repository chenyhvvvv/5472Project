import os

import pandas as pd

from models.base_model import BaseModel
from utils.networks import DeconvNet
import torch
import torch.nn as nn
from utils.losses import PoissonLoss

class Stitch3DModel(BaseModel):

    def __init__(self, opt, manager, stsc_dataset):
        super(Stitch3DModel, self).__init__(opt, manager)
        self.logger.info("======> Building Stitch3DModel")
        self.training_steps = self.opt.training.training_steps
        self.lr = self.opt.training.lr
        self.dataset = stsc_dataset
        # Build network
        if self.opt.model.distribution == "Poisson":

            self.network = DeconvNet(
                hidden_dims=[self.dataset.st_gene_num, *self.opt.model.arch.hidden_dims],
                n_celltypes=self.dataset.n_celltype,
                n_slices=self.dataset.n_slices,
                n_heads=self.opt.model.arch.n_heads,
                slice_emb_dim=self.opt.model.arch.slice_emb_dim,
                coef_fe=self.opt.model.coef_fe
            ).to(self.device)
            self.criterion = PoissonLoss(
                node_features = self.dataset.X,
                basis = self.dataset.basis,
                count_matrix = self.dataset.Y,
                library_size = self.dataset.lY,
                coef_fe = self.opt.model.coef_fe
            )
        else:
            raise NotImplementedError()

        self.optimizer = torch.optim.Adamax(self.network.parameters(), lr=self.lr)

    def _train_one_step(self):
        self.network.train()
        self.optimizer.zero_grad()
        node_feats_recon, beta, alpha, gamma = self.network(
            adj_matrix=self.dataset.A,
            node_feats=self.dataset.X,
            slice_labels=self.dataset.slice
        )
        loss = self.criterion(node_feats_recon, beta, alpha, gamma)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        for step in range(self.training_steps):
            loss = self._train_one_step()
            if step % self.opt.training.log_every == 0:
                self.logger.info(f"| Step {step} | Loss: {loss}")

            if step % self.opt.training.save_every == 0:
                self.logger.info(f"======> Save ckpt at iter {step}")
                self.save(os.path.join(self.manager.get_checkpoint_dir(), 'checkpoint_' + str(step) + '.pt'))
        self.save(os.path.join(self.manager.get_checkpoint_dir(), 'checkpoint_latest' + '.pt'))
        return None

    def save(self, path):
        """Save the model state
        """
        save_dict = {'model_state': self.network.state_dict(),
                     'optimizer_state': self.optimizer.state_dict()}
        torch.save(save_dict, path)

    def load(self, path):
        """Load a model state from a checkpoint file
        """
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])

    def eval(self):
        self.network.eval()
        self.Z = self.network.encoder(self.dataset.A, self.dataset.X)

        # Deconvolve
        _, self.beta, self.alpha, self.gamma = self.network(
            adj_matrix=self.dataset.A,
            node_feats=self.dataset.X,
            slice_labels=self.dataset.slice
        )

        # Make new dir to manager.run_dir for saving results
        output_path = os.path.join(self.manager.get_run_dir(), 'results')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Add learned representations to full ST adata object
        embeddings = self.Z.detach().cpu().numpy()
        cell_reps = pd.DataFrame(embeddings, index=self.dataset.adata_st.obs.index)
        self.dataset.adata_st.obsm['latent'] = cell_reps.loc[self.dataset.adata_st.obs_names, ].values
        cell_reps.to_csv(os.path.join(output_path, "representation.csv"))

        # Add deconvolution results to original anndata objects
        b = self.beta.detach().cpu().numpy()
        n_spots = 0
        adata_st_decon_list = []
        for i, adata_st_i in enumerate(self.dataset.adata_st_list_raw):
            adata_st_i.obs.index = adata_st_i.obs.index + "-slice%d" % i
            decon_res = pd.DataFrame(b[n_spots:(n_spots + adata_st_i.shape[0]), :],
                                     columns=self.dataset.celltypes)
            decon_res.index = adata_st_i.obs.index
            adata_st_i.obs = adata_st_i.obs.join(decon_res)
            n_spots += adata_st_i.shape[0]
            adata_st_decon_list.append(adata_st_i)

            decon_res.to_csv(os.path.join(output_path, "prop_slice%d.csv" % i))
            adata_st_i.write(os.path.join(output_path, "res_adata_slice%d.h5ad" % i))

        coor_3d = pd.DataFrame(data=self.dataset.adata_st.obsm['3D_coor'], index=self.dataset.adata_st.obs.index,
                               columns=['x', 'y', 'z'])
        coor_3d.to_csv(os.path.join(output_path, "3D_coordinates.csv"))

        return adata_st_decon_list