import io
import json
import jsonpickle
import logging
import time
import argparse
import os

def create_artefact_dirs(output_dir: str, random_seed : int) -> None:
    """
    Creates artefact directories if they do not already exist

    :param output_dir: the base directory
    :param random_seed: the random seed of the experiment
    :return: None
    """
    if not os.path.exists(output_dir + "/results/logs/" + str(random_seed) + "/"):
        os.makedirs(output_dir + "/results/logs/" + str(random_seed) + "/")
    if not os.path.exists(output_dir + "/results/plots/" + str(random_seed) + "/"):
        os.makedirs(output_dir + "/results/plots/" + str(random_seed) + "/")
    if not os.path.exists(output_dir + "/results/data/" + str(random_seed) + "/"):
        os.makedirs(output_dir + "/results/data/" + str(random_seed) + "/")
    if not os.path.exists(output_dir + "/results/hyperparameters/" + str(random_seed) + "/"):
        os.makedirs(output_dir + "/results/hyperparameters/" + str(random_seed) + "/")
    if not os.path.exists(output_dir + "/results/gifs/" + str(random_seed) + "/"):
        os.makedirs(output_dir + "/results/gifs/" + str(random_seed) + "/")
    if not os.path.exists(output_dir + "/results/tensorboard/" + str(random_seed) + "/"):
        os.makedirs(output_dir + "/results/tensorboard/" + str(random_seed) + "/")


def setup_logger(name: str, logdir: str, time_str = None):
    """
    Configures the logger for writing log-data of experiments

    :param name: name of the logger
    :param logdir: directory to save log files
    :param time_str: time string for file names
    :return: None
    """
    # create formatter
    formatter = logging.Formatter('%(asctime)s,%(message)s')
    # log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # log to file
    if time_str is None:
        time_str = str(time.time())
    fh = logging.FileHandler(logdir + "/" + time_str + "_" + name + ".log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    #logger.addHandler(ch)
    return logger