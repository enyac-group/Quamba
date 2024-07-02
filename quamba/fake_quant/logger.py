# --------------------------------------------------------
# Logger
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# --------------------------------------------------------

import os
import sys
import logging
import functools
from termcolor import colored


@functools.lru_cache()
def create_logger(name='', verbose=True):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
        colored('(%(filename)s %(lineno)d)', 'yellow') + \
        ': %(levelname)s %(message)s'

    #create console handlers for master process
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)
    return logger