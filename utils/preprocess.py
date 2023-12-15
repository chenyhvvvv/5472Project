import numpy as np
import scanpy as sc
import scipy
import anndata as ad
import pandas as pd
from sklearn.metrics import pairwise_distances

def preprocess(logger,
               adata_st_list_input, # list of spatial transcriptomics (ST) anndata objects
               adata_ref_input, # reference single-cell anndata object
               celltype_ref_col="celltype", # column of adata_ref_input.obs for cell type information
               sample_col=None, # column of adata_ref_input.obs for batch labels
               celltype_ref=None, # specify cell types to use for deconvolution
               n_hvg_group=500, # number of highly variable genes for reference anndata
               three_dim_coor=None, # if not None, use existing 3d coordinates in shape [# of total spots, 3]
               coor_key="spatial_aligned", # "spatial_aligned" by default
               rad_cutoff=None, # cutoff radius of spots for building graph
               rad_coef=1.1, # if rad_cutoff=None, rad_cutoff is the minimum distance between spots multiplies rad_coef
               slice_dist_micron=None, # pairwise distances in micrometer for reconstructing z-axis
               prune_graph_cos=False, # prune graph connections according to cosine similarity
               cos_threshold=0.5, # threshold for pruning graph connections
               c2c_dist=100, # center to center distance between nearest spots in micrometer
               ):

    adata_st_list = adata_st_list_input.copy()

    logger.info("======> Finding highly variable genes...")
    adata_ref = adata_ref_input.copy()
    adata_ref.var_names_make_unique()
    # Remove mt-genes
    adata_ref = adata_ref[:, np.array(~adata_ref.var.index.isna())
                          & np.array(~adata_ref.var_names.str.startswith("mt-"))
                          & np.array(~adata_ref.var_names.str.startswith("MT-"))]
    if celltype_ref is not None:
        if not isinstance(celltype_ref, list):
            raise ValueError("'celltype_ref' must be a list!")
        else:
            adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[celltype_ref_col].values.astype(str)], :]
    else:
        celltype_counts = adata_ref.obs[celltype_ref_col].value_counts()
        celltype_ref = list(celltype_counts.index[celltype_counts > 1])
        adata_ref = adata_ref[[(t in celltype_ref) for t in adata_ref.obs[celltype_ref_col].values.astype(str)], :]

    # Remove cells and genes with 0 counts
    sc.pp.filter_cells(adata_ref, min_genes=1)
    sc.pp.filter_genes(adata_ref, min_cells=1)

    # Concatenate ST adatas
    for i in range(len(adata_st_list)):
        adata_st_new = adata_st_list[i].copy()
        adata_st_new.var_names_make_unique()
        # Remove mt-genes
        adata_st_new = adata_st_new[:, (np.array(~adata_st_new.var.index.str.startswith("mt-"))
                                    & np.array(~adata_st_new.var.index.str.startswith("MT-")))]
        adata_st_new.obs.index = adata_st_new.obs.index + "-slice%d" % i
        adata_st_new.obs['slice'] = i
        if i == 0:
            adata_st = adata_st_new
        else:
            genes_shared = adata_st.var.index & adata_st_new.var.index
            adata_st = adata_st[:, genes_shared].concatenate(adata_st_new[:, genes_shared], index_unique=None)

    adata_st.obs["slice"] = adata_st.obs["slice"].values.astype(int)

    # Take gene intersection
    genes = list(adata_st.var.index & adata_ref.var.index)
    adata_ref = adata_ref[:, genes]
    adata_st = adata_st[:, genes]

    # Select hvgs
    adata_ref_log = adata_ref.copy()
    sc.pp.log1p(adata_ref_log)
    hvgs = select_hvgs(adata_ref_log, celltype_ref_col=celltype_ref_col, num_per_group=n_hvg_group)

    logger.info("======> %d highly variable genes selected." % len(hvgs))
    adata_ref = adata_ref[:, hvgs]

    logger.info("======> Calculate basis for deconvolution...")
    sc.pp.filter_cells(adata_ref, min_genes=1)
    sc.pp.normalize_total(adata_ref, target_sum=1)
    celltype_list = list(sorted(set(adata_ref.obs[celltype_ref_col].values.astype(str))))

    basis = np.zeros((len(celltype_list), len(adata_ref.var.index)))
    if sample_col is not None:
        sample_list = list(sorted(set(adata_ref.obs[sample_col].values.astype(str))))
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp_list = []
            for j in range(len(sample_list)):
                s = sample_list[j]
                tmp = adata_ref[(adata_ref.obs[celltype_ref_col].values.astype(str) == c) &
                                (adata_ref.obs[sample_col].values.astype(str) == s), :].X
                if scipy.sparse.issparse(tmp):
                    tmp = tmp.toarray()
                if tmp.shape[0] >= 3:
                    tmp_list.append(np.mean(tmp, axis=0).reshape((-1)))
            tmp_mean = np.mean(tmp_list, axis=0)
            if scipy.sparse.issparse(tmp_mean):
                tmp_mean = tmp_mean.toarray()
            logger.info("======> %d batches are used for computing the basis vector of cell type <%s>." % (len(tmp_list), c))
            basis[i, :] = tmp_mean
    else:
        for i in range(len(celltype_list)):
            c = celltype_list[i]
            tmp = adata_ref[adata_ref.obs[celltype_ref_col].values.astype(str) == c, :].X
            if scipy.sparse.issparse(tmp):
                tmp = tmp.toarray()
            basis[i, :] = np.mean(tmp, axis=0).reshape((-1))

    adata_basis = ad.AnnData(X=basis)
    df_gene = pd.DataFrame({"gene": adata_ref.var.index})
    df_gene = df_gene.set_index("gene")
    df_celltype = pd.DataFrame({"celltype": celltype_list})
    df_celltype = df_celltype.set_index("celltype")
    adata_basis.obs = df_celltype
    adata_basis.var = df_gene
    adata_basis = adata_basis[~np.isnan(adata_basis.X[:, 0])]

    logger.info("======> Preprocess ST data...")
    # Store counts and library sizes for Poisson modeling
    st_mtx = adata_st[:, hvgs].X.copy()
    if scipy.sparse.issparse(st_mtx):
        st_mtx = st_mtx.toarray()
    adata_st.obsm["count"] = st_mtx
    st_library_size = np.sum(st_mtx, axis=1)
    adata_st.obs["library_size"] = st_library_size

    # Normalize ST data
    sc.pp.normalize_total(adata_st, target_sum=1e4)
    sc.pp.log1p(adata_st)
    adata_st = adata_st[:, hvgs]
    if scipy.sparse.issparse(adata_st.X):
        adata_st.X = adata_st.X.toarray()

    # Build a graph for spots across multiple slices
    logger.info("======> Start building a graph...")

    # Build 3D coordinates
    if three_dim_coor is None:

        # The first adata in adata_list is used as a reference for computing cutoff radius of spots
        adata_st_ref = adata_st_list[0].copy()
        loc_ref = np.array(adata_st_ref.obsm[coor_key])
        pair_dist_ref = pairwise_distances(loc_ref)
        min_dist_ref = np.sort(np.unique(pair_dist_ref), axis=None)[1]

        if rad_cutoff is None:
            # The radius is computed base on the attribute "adata.obsm['spatial']"
            rad_cutoff = min_dist_ref * rad_coef
        logger.info("======> Radius for graph connection is %.4f." % rad_cutoff)

        # Use the attribute "adata.obsm['spatial_aligned']" to build a global graph
        if slice_dist_micron is None:
            loc_xy = pd.DataFrame(adata_st.obsm['spatial_aligned']).values
            loc_z = np.zeros(adata_st.shape[0])
            loc = np.concatenate([loc_xy, loc_z.reshape(-1, 1)], axis=1)
        else:
            if len(slice_dist_micron) != (len(adata_st_list) - 1):
                raise ValueError("The length of 'slice_dist_micron' should be the number of adatas - 1 !")
            else:
                loc_xy = pd.DataFrame(adata_st.obsm['spatial_aligned']).values
                loc_z = np.zeros(adata_st.shape[0])
                dim = 0
                for i in range(len(slice_dist_micron)):
                    dim += adata_st_list[i].shape[0]
                    loc_z[dim:] += slice_dist_micron[i] * (min_dist_ref / c2c_dist)
                loc = np.concatenate([loc_xy, loc_z.reshape(-1, 1)], axis=1)

    # If 3D coordinates already exists
    else:
        if rad_cutoff is None:
            raise ValueError("Please specify 'rad_cutoff' for finding 3D neighbors!")
        loc = three_dim_coor

    pair_dist = pairwise_distances(loc)
    G = (pair_dist < rad_cutoff).astype(float)

    if prune_graph_cos:
        pair_dist_cos = pairwise_distances(adata_st.X, metric="cosine") # 1 - cosine_similarity
        G_cos = (pair_dist_cos < (1 - cos_threshold)).astype(float)
        G = G * G_cos

    logger.info('======> %.4f neighbors per cell on average.' % (np.mean(np.sum(G, axis=1)) - 1))
    adata_st.obsm["graph"] = G
    adata_st.obsm["3D_coor"] = loc

    return adata_st, adata_basis
def select_hvgs(adata_ref, celltype_ref_col, num_per_group=200):
    sc.tl.rank_genes_groups(adata_ref, groupby=celltype_ref_col, method="t-test", key_added="ttest", use_raw=False)
    markers_df = pd.DataFrame(adata_ref.uns['ttest']['names']).iloc[0:num_per_group, :]
    genes = sorted(list(np.unique(markers_df.melt().value.values)))
    return genes