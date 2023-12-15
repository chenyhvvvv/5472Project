import sys
import os
sys.path.append("./")
from data import find_dataset_using_name
from utils.experiman import ExperiMan
from options import Options, HParams
import torch
import argparse

from data.stsc_dataset import STSCDataset
from models.stitch_3d_model import Stitch3DModel

def main(opt, manager):
    """Assume Single GPU"""
    assert torch.cuda.is_available(), "CPU training is not allowed."
    logger = manager.get_logger()

    logger.info(f"======> Single GPU Training")
    # Set up tensorboard`
    manager._third_party_tools = ('tensorboard',)
    manager._setup_third_party_tools()

    # Create dataset
    dataset = STSCDataset(opt, manager)
    logger.info("======> dataset [%s] was created" % type(dataset).__name__)
    logger.info(f"======> Total ST Spots: {dataset.X.shape[0]}")
    logger.info(f"======> Total Gene num: {dataset.st_gene_num}")
    logger.info(f"======> Total Cell Types: {dataset.n_celltype}")
    logger.info(f"======> Total Slices: {dataset.n_slices}")

    # Create model
    model = Stitch3DModel(opt, manager, dataset)
    model.train()
    model.eval()

if __name__ == "__main__":
    manager = ExperiMan(name='default')
    parser = manager.get_basic_arg_parser()
    opt = Options(parser).parse()  # get training options

    manager.setup(opt)
    opt = HParams(**vars(opt))

    main(opt, manager)