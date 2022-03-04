import sys
import wandb
from omegaconf import OmegaConf as omg
import torch
import numpy as np
from agent import Agent
from trainer import Trainer, reward_fn
from data_generator_v2 import DataGenerator, SimulatedAnnealing
import time
import os
import pandas as pd


np.random.seed(16)


def load_conf():
    """Quick method to load configuration (using OmegaConf). By default,
    configuration is loaded from the default config file (config.yaml).
    Another config file can be specific through command line.
    Also, configuration can be over-written by command line.

    Returns:
        OmegaConf.DictConfig: OmegaConf object representing the configuration.
    """
    default_conf = omg.create({"config" : "config.yaml"})

    sys.argv = [a.strip("-") for a in sys.argv]
    cli_conf = omg.from_cli()

    yaml_file = omg.merge(default_conf, cli_conf).config

    yaml_conf = omg.load(yaml_file)

    return omg.merge(default_conf, yaml_conf, cli_conf) 


def Adjacency(state):
    adj = []
    dis = []
    for j in range(20):
        dis.append([state[j][-2],state[j][-1],j])
    for j in range(20):
        f = []
        for r in range(len(dis)):
            f.append([(dis[r][0]-dis[j][0])**2+(dis[r][1]-dis[j][1])**2,r])
        f.sort(key=lambda x:x[0])
        y = []
        for r in range(4):
            y.append(f[r][1])
        y = to_categorical(y,num_classes=20)
        adj.append(y)
    return adj

def observation(state1,state2):
    state = []
    for j in range(20):
        state.append(np.hstack(((state1[j][0:11,0:11,1]-state1[j][0:11,0:11,5]).flatten(),state2[j][-1:-3:-1])))
    return state
