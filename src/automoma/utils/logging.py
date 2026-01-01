import logging

def setup_logger(
    name: str,
    level: int = logging.INFO,
    stream: bool = True,
    log_file: str | None = None,
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 防止重复打印

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(levelname)s]%(asctime)s: %(message)s",
        datefmt="%m%d-%H%M%S"
    )

    if stream:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

logger = setup_logger(__name__)