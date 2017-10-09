from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from contextlib import contextmanager
import pickle
import datetime

def generate_policy_from_pickle(picklefile):
    pass

def generate_dynamics_model(env,pol,activation):
    pass

def find_hyper_planes(dyn_m,pol,state_bounds=None,setofpoints=None):
    pass