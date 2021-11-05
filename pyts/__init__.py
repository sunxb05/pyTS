"""
Package containing Tensorfield Network tf.keras layers built using TF 2.0
"""
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL
logging.getLogger("tensorflow").setLevel(logging.FATAL)

__version__ = "0.0.1"
__author__ = "Xiaobo Sun"
__email__ = "sunxb05@gmail.com"
__description__ = (
    "Package containing pyTS built using Tensorflow 2"
)
