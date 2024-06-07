import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from WanPy.pythtb_Wannier import *
import WanPy.pythtb_Wannier as pythtb_Wannier 
from pythtb import *
import WanPy.models as models
import WanPy.plotting as plot

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sympy as sp 
import scipy
###########################

delta = 1
t0 = 0.4 # >= 0.25 in Chern phase
tprime = 0.5

model = models.chessboard(t0, tprime, delta).make_supercell([[2,0], [0,2]])

# For reference throughout
orbs = model.get_orb()
n_orb = model.get_num_orbitals()
n_occ = int(n_orb/2)
lat_vecs = model.get_lat() # lattice vectors

low_E_sites = np.arange(0, model.get_num_orbitals(), 2)
high_E_sites = np.arange(1, model.get_num_orbitals(), 2)

### Chern number
u_wfs_full = wf_array(model, [20, 20])
u_wfs_full.solve_on_grid([0, 0])
chern = u_wfs_full.berry_flux([i for i in range(n_occ)])/(2*np.pi)
print("Chern number: ", chern)
print(" ")

### Bands
k_path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
k_label = (r'$\Gamma $',r'$X$', r'$M$', r'$\Gamma $')
title = (rf"$C = {chern: .1f}$ | $\Delta = {delta},\ t_0 = {t0},\ t' = {tprime}$")

fig, ax = plot.plot_bands(
    model, k_path=k_path, k_label=k_label, sub_lat=True, red_lat_idx=high_E_sites, title=title, show=True)

### Projection trial wfs
omit_sites = 4 # tuple of site idices to exclude from lower energy (even) sites
tf_list = list(np.setdiff1d(low_E_sites, [omit_sites])) # delta on lower energy sites omitting the last site

### Band structure in tilde subspace
nk = 101
(k_vec, k_dist, k_node) = model.k_path(k_path, nk, report=False)

evals, u_path = model.solve_all(k_vec, eig_vectors=True)
u_path = np.transpose(u_path, axes=(1,0,2)) # [*nk, n, orb]
psi_path = get_bloch_wfs(model, u_path, k_vec)
psi_tilde_path = get_psi_tilde(psi_path, tf_list)
u_tilde_path = get_bloch_wfs(model, psi_tilde_path, k_vec, inverse=True) 

subspace = u_tilde_path
eigvals_sub, evecs_sub = diag_h_in_subspace(model, subspace, k_vec, ret_evecs=True)

# Interpolated bands
fig, ax = plot.plot_bands(
    model, k_path, evals=eigvals_sub, evecs=evecs_sub, k_label=k_label, 
    sub_lat=True, red_lat_idx=high_E_sites, title=title, show=False)

# Original bands
for n in range(evals.shape[0]):
    ax.plot(k_dist, evals[n], c='k', lw=1.5, zorder=0)

plt.show()

### Wannierizing
# 2D k-mesh
nkx = 16
nky = 16
Nk = nkx*nky

k_mesh = gen_k_mesh(nkx, nky, flat=False, endpoint=False)

u_wfs_Wan = wf_array(model, [nkx, nky])

for i in range(k_mesh.shape[0]):
    for j in range(k_mesh.shape[1]):
        u_wfs_Wan.solve_on_one_point(k_mesh[i,j], [i,j])

w0, psi_til_wan = Wannierize(model, u_wfs_Wan, tf_list, ret_psi_til=True)
u_tilde_wan = get_bloch_wfs(model, psi_til_wan, k_mesh, inverse=True)

print("Plotting Wannier function from single shot projection")
print(" ")
plot.plot_Wan(w0, 0, orbs, lat_vecs, plot_decay=True, show=True)

### Spreads
# Spread of unrotated energy eigenstates
M = k_overlap_mat(u_wfs_Wan, orbs=orbs) # [kx, ky, b, m, n]
spread, expc_rsq, expc_r_sq = spread_recip(model, M, decomp=True)

print("Spread from energy eigenstates")
print(rf"Spread from M_kb of u_nk = {spread[0]}")
print(rf"Omega_I from M_kb of u_nk = {spread[1]}")
print(rf"Omega_til from M_kb of u_nk = {spread[2]}")
print(" ")

# Spread of rotated (tilde) Bloch states corresponding to Wannier functions
M = k_overlap_mat(u_tilde_wan, orbs=orbs) # [kx, ky, b, m, n]
spread, expc_rsq, expc_r_sq = spread_recip(model, M, decomp=True)

print("Spread from tilde states (first projection)")
print(rf"Spread from M_kb of \tilde{{u_nk}} = {spread[0]}")
print(rf"Omega_I from M_kb of \tilde{{u_nk}} = {spread[1]}")
print(rf"Omega_til from M_kb of \tilde{{u_nk}} = {spread[2]}")
print(" ")

### Finding optimal subspace
outer_states = u_wfs_Wan._wfs[..., :n_occ, :] # energy window states
inner_states = u_tilde_wan # states spanning subspace of outer states
util_min_Wan = find_optimal_subspace(
    model, outer_states, inner_states, k_mesh=None, full_mesh=False, tol=2e-4, iter_num=4000)

# Spreads of optimal subspace
M = k_overlap_mat(util_min_Wan, orbs=orbs) # [kx, ky, b, m, n]
spread, expc_rsq, expc_r_sq = spread_recip(model, M, decomp=True)

print("Spread from states after minimizing omega_I")
print(rf"Spread from M_kb of \tilde{{u_nk}}_{{min}} = {spread[0]}")
print(rf"Omega_I from M_kb of \tilde{{u_nk}}_{{min}} = {spread[1]}")
print(rf"Omega_til from M_kb \tilde{{u_nk}}_{{min}} = {spread[2]}")
print(" ")

### Second projection (instead of find_min_unitary to find smooth gauge)
psi_til_min = get_bloch_wfs(model, util_min_Wan, k_mesh)
state_idx = list(range(psi_til_min.shape[2])) # specify which indices to compute overlap with
psi_til_til_min = get_psi_tilde(psi_til_min, tf_list, state_idx=state_idx)
u_til_til_min = get_bloch_wfs(model, psi_til_til_min, k_mesh, inverse=True)

M = k_overlap_mat(u_til_til_min, orbs=orbs) # [kx, ky, b, m, n]
spread, expc_rsq, expc_r_sq = spread_recip(model, M, decomp=True)

print("Spread from states after minimizing omega_I + 2nd projection")
print(rf"Spread from M_kb of \tilde\tilde{{u_nk}}_{{min}} = {spread[0]}")
print(rf"Omega_I from M_kb of \tilde\tilde{{u_nk}}_{{min}} = {spread[1]}")
print(rf"Omega_til from M_kb of \tilde\tilde{{u_nk}}_{{min}} = {spread[2]}")
print(" ")

### Finding optimal gauge choice
U, _ = find_min_unitary(model, M, iter_num=6000, eps=1e-3, print_=False)

u_min = np.zeros(u_til_til_min.shape, dtype=complex)
for kx in range(nkx):
    for ky in range(nky):
        for i in range(u_min.shape[2]):
            for j in range(u_min.shape[2]):
                u_min[kx, ky, i, :] += U[kx, ky, j, i] * u_til_til_min[kx, ky, j] 

psi_min = get_bloch_wfs(model, u_min, k_mesh, inverse=False)

# Spreads of maximally localized states 
M = k_overlap_mat(u_min, orbs=orbs) # [kx, ky, b, m, n]
spread, expc_rsq, expc_r_sq = spread_recip(model, M, decomp=True)

print("Spread from states after minimizing omega_I + 2nd projection + minimizing omega_tilde")
print(rf"Spread from M_kb of max_loc{{u_nk}} = {spread[0]}")
print(rf"Omega_I from M_kb of max_loc{{u_nk}} = {spread[1]}")
print(rf"Omega_til from M_kb of max_loc{{u_nk}} = {spread[2]}")
print(" ")

# Wannier funstions
print("Plotting maximally localized Wannier states")
w0 = DFT(psi_min)
plot.plot_Wan(w0, 0, orbs, lat_vecs, plot_decay=True, show=True)