# Quick dataset stats
from utils.helpers import get_dataloader
from argparse import Namespace

params = Namespace(
    data="MPIIGaze",
    dataset="mpiigaze",
    bins=90,
    binwidth=4.0,
    angle=180.0,
    batch_size=64,
    num_workers=4
)

loader = get_dataloader(params, mode="train")
print(f"Total samples: {len(loader.dataset):,}")
print(f"Total batches: {len(loader):,}")