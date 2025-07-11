import logging
import os


def init_logger(log_file=None, level=logging.INFO):
    """
    初始化日志系统
    :param log_file: 日志文件路径，如 None，则只输出到终端
    :param level: 日志等级，默认INFO
    """
    # 格式化日志内容
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(filename)s:%(lineno)d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger()
    logger.setLevel(level)

    # 清除已有的handler，防止重复
    if logger.hasHandlers():
        logger.handlers.clear()

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


def get_logger(name=None):
    """
    获取logger对象
    :param name: logger名
    :return: logger实例
    """
    return logging.getLogger(name)

# 初始化日志，输出到文件和终端
init_logger("logs/app.log", level=logging.ERROR)
my_logger = get_logger()