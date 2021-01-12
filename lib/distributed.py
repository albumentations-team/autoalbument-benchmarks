def check_if_main_worker(use_distributed, rank):
    return not use_distributed or rank == 0
