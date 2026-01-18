import os
import logging

logs = set()

def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return

    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler() # StreamHandler类可以被用来将日志信息输出到如控制台或文件这样的输出流
    ch.setLevel(level) # 指定被处理的信息级别，低于lel级别的信息将被忽略

    fh = logging.FileHandler('out.txt')
    fh.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

