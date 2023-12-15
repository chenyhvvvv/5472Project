import numpy as np
import scipy
import torch

from data.base_dataset import BaseDataset
import anndata as ad
import pickle

from utils.alignment import align_slices
from utils.preprocess import preprocess


class STSCDataset(BaseDataset):
    def __init__(self, opt, manager):
        BaseDataset.__init__(self, opt, manager)
        # Load data after cleaning
        with open(f"{opt.dataset.preprocess_pkl_loc}st_data.pickle", 'rb') as f:
            self.adata_st_list_raw = pickle.load(f)
        with open(f"{opt.dataset.preprocess_pkl_loc}sc_data.pickle", 'rb') as f:
            self.adata_ref = pickle.load(f)
        with open(f"{opt.dataset.preprocess_pkl_loc}para_dict.pickle", 'rb') as f:
            self.data_para_dict = pickle.load(f)

        # Setup, only support cuda device
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            raise EnvironmentError()

        # Align data: with extra parameters from opt and data_para_dict
        self.aligned_st = align_slices(self.logger, self.adata_st_list_raw, **self.opt.dataset.align,
                                      **self.data_para_dict["align"])
        # Preprocess data: with extra parameters from opt and data_para_dict
        self.adata_st, self.adata_basis = preprocess(self.logger, self.aligned_st, self.adata_ref, **self.opt.dataset.preprocess,
                                           **self.data_para_dict["preprocess"])

        # Get the data
        if scipy.sparse.issparse(self.adata_st.X):
            self.X = torch.from_numpy(self.adata_st.X.toarray()).float().to(self.device)
        else:
            self.X = torch.from_numpy(self.adata_st.X).float().to(self.device)
        self.A = torch.from_numpy(np.array(self.adata_st.obsm["graph"])).float().to(self.device)
        self.Y = torch.from_numpy(np.array(self.adata_st.obsm["count"])).float().to(self.device)
        self.lY = torch.from_numpy(np.array(self.adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(self.device)
        self.slice = torch.from_numpy(np.array(self.adata_st.obs["slice"].values)).long().to(self.device)
        self.basis = torch.from_numpy(np.array(self.adata_basis.X)).float().to(self.device)

        # Other properties
        self.celltypes = list(self.adata_basis.obs.index)
        self.st_gene_num = self.adata_st.shape[1]
        self.n_celltype = self.adata_basis.shape[0]
        self.n_slices = len(sorted(set(self.adata_st.obs["slice"].values)))
    def __len__(self):
        # Not used
        return 0

    def __getitem__(self, index):
        # Not used
        return 0