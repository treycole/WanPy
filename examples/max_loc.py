import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from WanPy.WanPy import *
import WanPy.WanPy as WanPy 
from pythtb import *
import WanPy.models as models
import WanPy.plotting as plot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sympy as sp 
import scipy
###########################

# Checkerboard model
# delta = 1
# t0 = 0.4 # >= 0.25 in Chern phase
# tprime = 0.5
# model = models.chessboard(t0, tprime, delta).make_supercell([[2,0], [0,2]])

# Haldane model
delta = 1
t = -1
t2 = 0.2 # >= 0.2 in Chern phase
model = models.Haldane(delta, t, t2).make_supercell([[2,0], [0,2]])

orbs = model.get_orb()
n_orb = model.get_num_orbitals()
n_occ = int(n_orb/2)
lat_vecs = model.get_lat() # lattice vectors

# Reporting net Chern number of occupied manifold
u_wfs_full = wf_array(model, [20, 20])
u_wfs_full.solve_on_grid([0, 0])
chern = u_wfs_full.berry_flux([i for i in range(n_occ)])/(2*np.pi)
print(f"Chern number: {chern: .3f}")

# get Bloch eigenstates on 2D k-mesh for Wannierization (exclude endpoints)
nkx = 20
nky = 20
Nk = nkx*nky
k_mesh = gen_k_mesh(nkx, nky, flat=False, endpoint=False)
u_wfs_Wan = wf_array(model, [nkx, nky])
for i in range(k_mesh.shape[0]):
    for j in range(k_mesh.shape[1]):
        u_wfs_Wan.solve_on_one_point(k_mesh[i,j], [i,j])

# Wannierization via single-shot projection
low_E_sites = np.arange(0, model.get_num_orbitals(), 2)
high_E_sites = np.arange(1, model.get_num_orbitals(), 2)
omit_sites = 4
tf_list = list(np.setdiff1d(low_E_sites, [omit_sites])) # delta on lower energy sites omitting the last site

w0, psi_til_wan = Wannierize(model, u_wfs_Wan, tf_list, ret_psi_til=True)
u_tilde_wan = get_bloch_wfs(model, psi_til_wan, k_mesh, inverse=True)
# plot.plot_Wan(w0, 0, orbs, lat_vecs, plot_decay=True, show=True)

# outer window of entangled bands is full occupied manifold
outer_states = u_wfs_Wan._wfs[..., :n_occ, :]
w0_max_loc = max_loc_Wan(model, u_wfs_Wan, tf_list, outer_states, 
        iter_num_omega_i=2000, iter_num_omega_til=5000,
        state_idx=None, print_=True, return_uwfs=False, eps=2e-3, report=True
        )

Wan_idx = 0
plot.plot_Wan(w0_max_loc, Wan_idx, orbs, lat_vecs, plot_decay=True, show=True)
