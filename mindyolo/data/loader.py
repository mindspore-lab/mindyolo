"""
Create dataloader
"""
import multiprocessing

import cv2
import mindspore.dataset as de

from mindyolo.utils import logger

__all__ = ["create_loader"]


def create_loader(
    dataset,
    batch_collate_fn,
    column_names_getitem,
    column_names_collate,
    batch_size,
    epoch_size=1,
    rank=0,
    rank_size=1,
    num_parallel_workers=8,
    shuffle=True,
    drop_remainder=False,
    python_multiprocessing=False,
):
    r"""Creates dataloader.

    Applies operations such as transform and batch to the `ms.dataset.Dataset` object
    created by the `create_dataset` function to get the dataloader.

    Args:
        dataset (COCODataset): dataset object created by `create_dataset`.
        batch_size (int or function): The number of rows each batch is created with. An
            int or callable object which takes exactly 1 parameter, BatchInfo.
        drop_remainder (bool, optional): Determines whether to drop the last block
            whose data row number is less than batch size (default=False). If True, and if there are less
            than batch_size rows available to make the last batch, then those rows will
            be dropped and not propagated to the child node.
        num_parallel_workers (int, optional): Number of workers(threads) to process the dataset in parallel
            (default=None).
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker processes. This
            option could be beneficial if the Python operation is computational heavy (default=False).

    Returns:
        BatchDataset, dataset batched.
    """
    cores = multiprocessing.cpu_count()
    num_parallel_workers = min(int(cores / rank_size), num_parallel_workers)
    logger.info(f"Dataloader num parallel workers: [{num_parallel_workers}]")
    de.config.set_seed(1236517205 + rank * num_parallel_workers)
    if rank_size > 1:
        ds = de.GeneratorDataset(
            dataset,
            column_names=column_names_getitem,
            num_parallel_workers=min(8, num_parallel_workers),
            shuffle=shuffle,
            python_multiprocessing=python_multiprocessing,
            num_shards=rank_size,
            shard_id=rank,
        )
    else:
        ds = de.GeneratorDataset(
            dataset,
            column_names=column_names_getitem,
            num_parallel_workers=min(32, num_parallel_workers),
            shuffle=shuffle,
            python_multiprocessing=python_multiprocessing,
        )
    ds = ds.batch(
        batch_size, per_batch_map=batch_collate_fn,
        input_columns=column_names_getitem, output_columns=column_names_collate, drop_remainder=drop_remainder
    )
    ds = ds.repeat(epoch_size)

    return ds
