import numpy as np
from pythtb import *

# For testing purposes
def chessboard(t0, tprime, delta):
    # define lattice vectors
    lat=[[1.0, 0.0], [0.0, 1.0]]
    # define coordinates of orbitals
    orb=[[0.0, 0.0], [0.5, 0.5]]

    # make two dimensional tight-binding checkerboard model
    model = tb_model(2, 2, lat, orb)

    # set on-site energies
    model.set_onsite([-delta, delta], mode='reset')

    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])
    model.set_hop(-t0, 0, 0, [1, 0], mode='reset')
    model.set_hop(-t0, 0, 0, [0, 1], mode='reset')
    model.set_hop(t0, 1, 1, [1, 0], mode='reset')
    model.set_hop(t0, 1, 1, [0, 1], mode='reset')

    model.set_hop(tprime, 1, 0, [1, 1], mode='reset')
    model.set_hop(tprime*1j, 1, 0, [0, 1], mode='reset')
    model.set_hop(-tprime, 1, 0, [0, 0], mode='reset')
    model.set_hop(-tprime*1j, 1, 0, [1, 0], mode='reset')

    return model

# Ultimately we will do all this on a 2D k-mesh, not a path in k-space, so some details would need to change.
# We are going to use wfs_psi, not wfs, in everything else we do for the Wannier constructions. 

def get_orb_phases(model, k_vec):
  """
  Introduces e^i{k.tau} factors

  Args:
      model (pythb.tb_model): PythTB model
      k_vec (np.array): k space grid

  Returns:
    orb_phases (np.array): array of phases at each k value
  """
  orb = model.get_orb()   # numpy array in order [orbital, reduced coord]
  per_dir = model._per    # list of periodic dimensions

  # slice second dimension to only keep only periodic dimensions in orb
  per_orb = orb[:, per_dir]
  
  # compute a list of phase factors [k_val, orbital]
  orb_phases = np.array([
    [np.exp(1j* 2 * np.pi * per_orb[i] @ k_vec[j] )
     for i in range(per_orb.shape[0])]
     for j in range(k_vec.shape[0])]) 
  
  return orb_phases  # 1D numpy array of dimension norb

def get_bloch_wfs(model, wfs, kvec):
  """
  Change the cell periodic wfs to Bloch wfs
  
  Args:
    model (pythb.tb_model): PythTB model
    wfs (np.array): cell periodic wfs
    kvec (np.array): path in k space
      
  Returns:
    wfs_psi (np.array): wfs with orbitals multiplied by proper phase factor

  """
  nk = kvec.shape[0]
  norb = model.get_num_orbitals()

  wfs_psi = np.zeros((nk, norb, norb), dtype=complex)  # [k, nband, norb]
  phases = get_orb_phases(model, kvec)

  for k in range(nk):
    for i in range(norb):
      wfs_psi[k, :, i] = wfs[k][:][i] * phases[k, i]

  return wfs_psi


def set_trial_function(tf_list, norb):
  """
  Args:
      tf_list: list[int | tuple]
        list of numbers or tuples defining either the integer site 
        of the trial function (delta) or the tuples of sites and 
        amplitudes
      norb: int
        number of orbitals in the primative unit cell

  Returns:
      tfs (num_tf x norb np.array): 2 dimensional array of trial functions 
  """
 
  # number of trial functions to define
  num_tf = len(tf_list)

  # initialize array containing tfs = "trial functions"
  tfs = np.zeros([num_tf, norb], dtype=complex)

  for j, tf in enumerate(tf_list):
    if isinstance(tf, int): 
      # We only have a trial function on one site
      tfs[j, tf] = 1
    elif isinstance(tf, list):
      # Must be list of tuples of the form (site, amplitude)
      for site, amp in tf:
        tfs[j, site] = amp
      # normalizing 
      tfs[j,:] /= sum(abs(tfs[j, :]))
    else:
      raise TypeError("tf_list is not of apporpriate type")

  # return numpy array containing trial functions
  return tfs # tfs in order[trial funcs, orbitals]


# %%
def tf_overlap_mat(model, wfs, kvec, tfs, n_occ):
    """_summary_

    Args:
        model: pythtb.model
        wfs: np.array
            cell periodic wfs
        tfs: np.array
            array of trial wfs
        n_occ: int
            number of occupied bands

    Retruns:
        A: np.array
            overlap matrix
    """

    ntfs = tfs.shape[0]
    nk = kvec.shape[0]

    A = np.zeros((nk, n_occ, ntfs),dtype=complex)
    wfs_bloch = get_bloch_wfs(model, wfs, kvec)
    
    for k in range(nk):
        for n in range(n_occ):
            for j in range(ntfs):
                A[k, n, j] = np.vdot(wfs_bloch[k, n, :], tfs[j, :])

    return A

# %%
delta = 1
t0 = .1
tprime = .1

model = chessboard(t0, tprime, delta)

# generate k-point path and labels
nk = 51
path = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.5], [0.0, 0.0]]
(k_vec, k_dist, k_node) = model.k_path(path, nk, report=False)

wfs = wf_array(model, [nk])

for k in range(nk):
    wfs.solve_on_one_point([k_vec[k,0], k_vec[k,1]], [k])

phases = get_orb_phases(model, k_vec)
wfs_bloch = get_bloch_wfs(model, wfs, k_vec)

norb = 2
tf_list = [[(0, 0), (1, 1)], [(0, 1), (1, 1)], [(0, 1), (1, 0)]]

tfs = set_trial_function(tf_list, norb)

A = tf_overlap_mat(model, wfs, k_vec, tfs, 1)

print("Trial fxn overlap matrix A: ", A)
