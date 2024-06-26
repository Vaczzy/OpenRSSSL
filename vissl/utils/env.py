# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from vissl.config import AttrDict


def set_env_vars(local_rank: int, node_id: int, cfg: AttrDict):
    """
    Set some environment variables like total number of gpus used in training,
    distributed rank and local rank of the current gpu, whether to print the
    nccl debugging info and tuning nccl settings.
    """
    os.environ["WORLD_SIZE"] = str(
        cfg.DISTRIBUTED.NUM_NODES * cfg.DISTRIBUTED.NUM_PROC_PER_NODE
    )
    dist_rank = cfg.DISTRIBUTED.NUM_PROC_PER_NODE * node_id + local_rank
    os.environ["RANK"] = str(dist_rank)
    # os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(cfg.DISTRIBUTED.PROC_ID) # Support to select gpu //Vaczzy 2024.06.11 default: 0
    if cfg.DISTRIBUTED.NCCL_DEBUG:
        os.environ["NCCL_DEBUG"] = "INFO"
    if cfg.DISTRIBUTED.NCCL_SOCKET_NTHREADS:
        logging.info(
            f"local_rank: {local_rank}, "
            f"using NCCL_SOCKET_NTHREADS: {cfg.DISTRIBUTED.NCCL_SOCKET_NTHREADS}"
        )
        os.environ["NCCL_SOCKET_NTHREADS"] = str(cfg.DISTRIBUTED.NCCL_SOCKET_NTHREADS)


def setup_path_manager():
    """
    Registering the right options for the g_pathmgr:
    Override this function in your build system to
    support different distributed file system
    """
    pass


def print_system_env_info(current_env):
    """
    Print information about user system environment where VISSL is running.
    """
    keys = list(current_env.keys())
    keys.sort()
    for key in keys:
        logging.info("{}:\t{}".format(key, current_env[key]))


def get_machine_local_and_dist_rank():
    """
    Get the distributed and local rank of the current gpu.
    """
    # local_rank = int(os.environ.get("LOCAL_RANK",0))  # Support to select gpu //Vaczzy 2024.06.11
    local_rank = int(os.environ.get("LOCAL_RANK"))
    distributed_rank = int(os.environ.get("RANK", 0))
    return local_rank, distributed_rank
