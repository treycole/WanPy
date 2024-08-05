import sys
import os
sys.path.append("../WanPy")

from pythTB_wan import Wannier, Bloch, K_mesh
import models as models
import numpy as np
from pythtb import *
import matplotlib.pyplot as plt

import pickle

delta = 1
t0_list = np.linspace(0.1, 0.4, 50)
tprime = 0.5

omega_i = np.zeros(t0_list.shape[0])
omega_til = np.zeros(t0_list.shape[0])
chern = np.zeros(t0_list.shape[0])

WF_dict = {}

n_super_cell = 2
nkx, nky = 30, 30
n_tf = 2

for idx, t0 in enumerate(t0_list):
    print(idx)

    model = models.chessboard(t0, tprime, delta).make_supercell([[n_super_cell, 0], [0, n_super_cell]])

    # random low E twfs
    low_E_sites = np.arange(0, model.get_num_orbitals(), 2)
    tf_list = np.random.choice(low_E_sites, n_tf, replace=False) # ["random", n_tf]

    # specific low E twfs
    # omit_sites = 6, 4
    # tf_list = list(np.setdiff1d(low_E_sites, [omit_sites])) # delta on lower energy sites omitting the last site

    # random twfs
    # omit_num = 1
    # n_tfs = n_occ - omit_num
    # tf_list = ["random", n_tfs]

    WFs = Wannier(model, [nkx, nky])
    WFs.Wannierize(tf_list)
    WFs.max_loc(
        iter_num_omega_i=10000, iter_num_omega_til=50000, 
        eps=1e-3, tol_omega_i=1e-3, tol_omega_til=1e-3,
        grad_min=1, verbose=True)
    
    WF_dict[t0] = WFs

    file_name = f"data/n_tfs_{n_tf}_{n_super_cell}_prim_cells_{nkx}x{nky}_k_mesh_fxn_of_t0"
    np.save(f"{file_name}_WFs.npy", WF_dict)