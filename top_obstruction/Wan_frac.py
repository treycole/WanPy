import sys
import os
sys.path.append("../WanPy")

from pythTB_wan import Wannier, Bloch, K_mesh
import models as models
import numpy as np
from pythtb import *
import matplotlib.pyplot as plt

import pickle

# delta = 1
# t0 = 0.4
# tprime = 0.5
# n_super_cell = 3
# model = models.chessboard(t0, tprime, delta).make_supercell([[n_super_cell, 0], [0, n_super_cell]])

delta = 1
t = 1
t2 = -0.3

n_super_cell = 5
model = models.Haldane(delta, t, t2).make_supercell([[n_super_cell, 0], [0, n_super_cell]])

low_E_sites = np.arange(0, model.get_num_orbitals(), 2)
n_orb = model.get_num_orbitals()
n_occ = int(n_orb/2)

u_wfs_full = wf_array(model, [20, 20])
u_wfs_full.solve_on_grid([0, 0])
chern = u_wfs_full.berry_flux([i for i in range(n_occ)])/(2*np.pi)

save_name = f'C={chern:.1f}_Delta={delta}_t={t}_t2={t2}'
name = f'Wan_frac_{save_name}_n_occ={n_occ}'

sv_dir = 'data'
if not os.path.exists(sv_dir):
    os.makedirs(sv_dir)

n_tfs = np.array([i for i in range(n_occ-1, 0, -1)])
Wan_frac = n_tfs/n_occ
WFs_n_tfs = {}

print(name)

for idx, n_tf in enumerate(n_tfs):
    print(Wan_frac[idx])

    tf_list = np.random.choice(low_E_sites, n_tf, replace=False) # ["random", n_tf]
    WFs = Wannier(model, [20, 20])
    WFs.single_shot(tf_list)

    WFs.ss_maxloc(
        verbose=True, iter_num_omega_i=20000, iter_num_omega_til=50000, 
        tol_omega_i=1e-3, tol_omega_til=1e-3, grad_min=1e-1, eps=5e-4
        )
    
    WFs_n_tfs[n_tf] = WFs

    np.save(f"{sv_dir}/{name}_WFs.npy", WFs_n_tfs)