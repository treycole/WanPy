from pythtb import *
from WanPy.pythtb_Wannier import *
import WanPy.models as models
import WanPy.plotting as plot

import numpy as np
#########################################

# model
delta = 1
t0 = 0.2
tprime = 0.5

model = models.chessboard(t0, tprime, delta).make_supercell([[2,0], [0,2]])

lat_vecs = model.get_lat() # lattice vectors
orbs = model.get_orb() # orbital vectors
n_orb = model.get_num_orbitals() # number of orbitals
n_occ = int(n_orb/2) # number of occupied states

low_E_sites = np.arange(0, n_orb, 2)
high_E_sites = np.arange(1, n_orb, 2)

# 2D k-mesh
nkx = 20
nky = 20
Nk = nkx*nky

# eigenstates
u_wfs_full = wf_array(model, [nkx, nky])
u_wfs_full.solve_on_grid([0, 0])

# Chern number
chern = u_wfs_full.berry_flux([i for i in range(n_occ)])/(2*np.pi)
print("Chern number: ", chern)
print(" ")

############  Band plot  ####################

k_path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
k_label = (r'$\Gamma $',r'$X$', r'$M$', r'$\Gamma $')

plot.plot_bands(
    model, k_path=k_path, k_label=k_label, sub_lat=True, red_lat_idx=high_E_sites, show=True)

######### Wannierizing ##############

u_wfs_Wan = wf_array(model, [nkx, nky])

k_mesh = gen_k_mesh(nkx, nky, flat=False, endpoint=False) # excl endpoints
for i in range(k_mesh.shape[0]):
    for j in range(k_mesh.shape[1]):
        u_wfs_Wan.solve_on_one_point(k_mesh[i,j], [i,j])

# trial wfs

omit_site = None
tf_list = list(np.setdiff1d(low_E_sites, [omit_site])) # delta on lower energy sites omitting site specified
n_tfs = len(tf_list) 

# Wannier fxns in home unit cell 
w0 = Wannierize(model, u_wfs_Wan, tf_list)

######## Plotting #######
idx = 0
title = ( rf"$C = {chern: .1f}$ | $\Delta = {delta},\ t_0 = {t0},\ t' = {tprime}$"
             "\n" 
             rf"Trial fxns on sites {tf_list}")
save_name = f'w_{idx}_scatter_C={chern:.1f}_Delta={delta}_t0={t0}_tp={tprime}_tfxs={tf_list}.png'

plot.plot_Wan(w0, idx, orbs, lat_vecs, title=title, plot_phase=True, plot_decay=True, fit_rng=[7, 20], show=True)

######## Spreads ########

### Real space 
spread, r_n, rsq_n  = spread_real(model, w0, decomp=True)

print(f"Real space spread: {spread[0]: .8f}")
print(f"Real space <r^2> = {np.sum(rsq_n): .8f}")
print(f"Real space <r>^2 = {np.sum([np.vdot(r_n[n,:], r_n[n,:]) for n in range(r_n.shape[0])]): .8f}")
print(rf"Real space Omega_I = {spread[1]: .8f}")
print(rf"Real space \tilde{{Omega}} = {spread[2]: .8f}")
print(rf"Omega_I + \tilde{{Omega}} = {spread[1] + spread[2]: .8f}" )
print(" ")

### Reciprocal Space
# getting cell periodic psi_tildes
psi_wfs = get_bloch_wfs(model, u_wfs_Wan, k_mesh)
psi_tilde = get_psi_tilde(psi_wfs, tf_list)
u_tilde = get_bloch_wfs(model, psi_tilde, k_mesh, inverse=True)

M = k_overlap_mat(u_tilde, orbs=orbs)
spread, r_n, rsq_n  = spread_recip(model, M, decomp=True)

print(f"Reciprocal space spread: {spread[0]: .8f}")
print(f"Reciprocal space <r^2> = {np.sum(rsq_n): .8f}")
print(f"Reciprocal space <r>^2 = {np.sum([np.vdot(r_n[n,:], r_n[n,:]) for n in range(r_n.shape[0])]): .8f}")
print(rf"Reciprocal space Omega_I = {spread[1]: .8f}")
print(rf"Reciprocal space \tilde{{Omega}} = {spread[2]: .8f}")
print(rf"Omega_I + \tilde{{Omega}} = {spread[1] + spread[2]: .8f}" )