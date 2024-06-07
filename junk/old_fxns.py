import numpy as np
from pythtb import *
from itertools import product

# used for testing purposes

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


def gen_k_mesh(*nks, centered=False, flat=True):
    if centered:
        end_pts = [-0.5, 0.5]
    else:
        end_pts = [0, 1]

    k_vals = [np.linspace(end_pts[0], end_pts[1], nk, endpoint=False) for nk in nks]
    mesh = np.array(list(product(*k_vals)))

    if not flat:
        return mesh.reshape(*[nk for nk in nks], len(nks))

    return mesh


def get_orb_phases(model, k_vec, inverse=False):
  """
  Introduces e^i{k.tau} factors

  Args:
      model (pythb.tb_model): PythTB model
      k_vec (np.array): k space grid (assumes flattened)
      inverse (boolean): whether to get cell periodic (True) or Bloch (False) wfs

  Returns:
    orb_phases (np.array): array of phases at each k value
  """
  lam = (-1)**inverse  # overall minus if getting cell periodic from Bloch
  orb = model.get_orb()   # numpy array in order [orbital, reduced coord]
  per_dir = model._per    # list of periodic dimensions
  # slice second dimension to only keep only periodic dimensions in orb
  per_orb = orb[:, per_dir]
 
  # compute a list of phase factors [k_val, orbital]
  wf_phases = np.array(np.exp(lam * 1j* 2 * np.pi * per_orb @ k_vec.T), dtype=complex).T

  # # compute a list of phase factors [k_val, orbital]
  # orb_phases = np.array([
  #   [np.exp(lam*1j* 2 * np.pi * per_orb[i] @ k_vec[j] )
  #    for i in range(per_orb.shape[0])]
  #    for j in range(k_vec.shape[0])], dtype=complex) 

  # assert(np.allclose(orb_phases, wf_phases)) # True

  return wf_phases  # 1D numpy array of dimension norb


def get_bloch_wfs(model, u_wfs, k_mesh):
  """
  Change the cell periodic wfs to Bloch wfs
  
  Args:
    model (pythtb.tb_model): PythTB model
    wfs (pythtb.wf_array): cell periodic wfs [k, nband, norb]
    kvec (np.array): path in k space
      
  Returns:
    wfs_psi: np.array
      wfs with orbitals multiplied by proper phase factor

  """

  nks = [*u_wfs._mesh_arr] # size of mesh in each direction
  norb = model.get_num_orbitals() # number of orbitals

  psi_wfs = np.zeros((*nks, norb, norb), dtype=complex)  # [*nk_i, nband, norb]
  # Phases come in a list flattened over k space
  # Needs be reshaped to match k indexing of wf_array 
  phases = get_orb_phases(model, k_mesh).reshape(*nks, norb) 

  # important to note the subtle difference in wfs[k][:][i] and
  # wfs[k][:, i]
  if len(nks) == 1:
     for k in range(nks[0]):
      for i in range(norb):
        psi_wfs[k, :, i] = u_wfs[k][:, i] * phases[k, i]

  elif len(nks) == 2:
    for kx in range(nks[0]):
      for ky in range(nks[1]):
        for i in range(norb):
          psi_wfs[kx, ky, :, i] = u_wfs[kx, ky][:, i] * phases[kx, ky, i] 

  elif len(nks) == 3:
    for kx in range(nks[0]):
      for ky in range(nks[1]):
        for kz in range(nks[2]):
          for i in range(norb):
            psi_wfs[kx, ky, kz, :, i] = u_wfs[kx, ky, kz][:, i] * phases[kx, ky, kz, i] 

  return psi_wfs


def tf_overlap_mat(model, k_mesh, u_wfs, tfs, n_occ):
    """_summary_

    Args:
        model: pythtb.model
        k_mesh: np.array
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
    nks = [*u_wfs._mesh_arr]
    ntfs = tfs.shape[0]
   
    psi_wfs = get_bloch_wfs(model, u_wfs, k_mesh)
    A = np.zeros((*nks, n_occ, ntfs), dtype=complex)

    if len(nks) == 1: # 1d k path
        for k in range(nks[0]):
            for n in range(n_occ):
                for j in range(ntfs):
                    A[k, n, j] = np.vdot(psi_wfs[k, n, :], tfs[j, :])
    
    elif len(nks) == 2: # 2d k path
        for kx in range(nks[0]):
            for ky in range(nks[1]):
                for n in range(n_occ):
                    for j in range(ntfs):
                        A[kx, ky, n, j] = np.vdot(psi_wfs[kx, ky, n, :], tfs[j, :])

    elif len(nks) == 3: # 3d k path
        for kx in range(nks[0]):
            for ky in range(nks[1]):
                for kz in range(nks[2]):
                    for n in range(n_occ):
                        for j in range(ntfs):
                            A[kx, ky, kz, n, j] = np.vdot(psi_wfs[kx, ky, kz, n, :], tfs[j, :])

    A2 = np.zeros((*nks, n_occ, ntfs), dtype=complex)

    # .transpose(*([i for i in range(len(nks))] +  [len(nks)+1, len(nks)]))
    for n in range(n_occ):
        for j in range(ntfs):
            A2[..., n, j] = (
                psi_wfs.conj()[..., n, :] @ tfs[j, :].T
            )

    print(np.allclose(A, A2))

    return A

def get_psi_tilde(psi_wf, A):
    nks = psi_wf.shape[:-2]
    n_orb = psi_wf.shape[-2]
    ntfs = A.shape[-1]
    n_occ = A.shape[-2]

    V, S, Wh = np.linalg.svd(A, full_matrices=False)
    M = V @ Wh
    psi_tilde = np.zeros((*nks, ntfs, n_orb), dtype=complex)  # [*nk_i, nband, norb]

    if len(nks) == 1:
        for k in range(nks[0]):
            for n in range(ntfs):
                psi_tilde[k, n, :] = np.sum(
                    *[psi_wf[k, m, :] * M[k, m, n] for m in range(n_occ)], axis=0
                    )

    if len(nks) == 2:
        for kx in range(nks[0]):
            for ky in range(nks[1]):
                for n in range(ntfs):
                    psi_tilde[kx, ky, n, :] = np.sum(
                        [psi_wf[kx, ky, m, :] * M[kx, ky, m, n] for m in range(n_occ)], axis=0
                        )
    
    if len(nks) == 3:
        for kx in range(nks[0]):
            for ky in range(nks[1]):
                for kz in range(nks[2]):
                    for n in range(ntfs):
                        psi_tilde[kx, ky, kz, n, :] = np.sum(
                            *[psi_wf[kx, ky, kz, m, :] * M[kx, ky, kz, m, n] for m in range(n_occ)], axis=0
                            )
                        
    psi_tilde_2 = M.transpose(0, 1, 3, 2) @ psi_wf[..., :n_occ, :]

    print(np.allclose(psi_tilde, psi_tilde_2))
    
    return psi_tilde


def set_trial_function(tf_list, norb):
  """
  Args:
      tf_list: list[int | tuple]
        list of numbers or tuples defining either the integer site 
        of the trial function (delta) or the tuples (site, amplitude)
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
      tfs[j,:] /= np.sqrt(sum(abs(tfs[j, :])))
    else:
      raise TypeError("tf_list is not of apporpriate type")

  # return numpy array containing trial functions
  return tfs # tfs in order[trial funcs, orbitals]


def SVD_A(A):
    # SVD on last 2 axes by default
    return np.linalg.svd(A)

def DFT(psi_tilde):
    Rn = np.fft.fftn(psi_tilde, axes = (psi_tilde.shape[:-2]), norm='forward')
    return Rn

# These two FT methods are equivalent
def DFT_man(psi_tilde):
    Rn = np.zeros(psi_tilde.shape, dtype=complex)
    nkx = psi_tilde.shape[0]
    nky = psi_tilde.shape[1]

    for kx in range(nkx):
        for ky in range(nky):
            for band in range(psi_tilde.shape[2]):
                Rn[kx, ky, band, :] = (nkx*nky)**(-1)*(
                    np.sum(
                        [psi_tilde[m, n, band, :] * np.exp(-1j*2*np.pi * ((m*kx/nkx) + (n*ky/nky)) ) 
                          for n in range(0, nky) for m in range(0, nkx)], axis=0) 
                          )
    return Rn

# # Compute the rectangular A matrix
# # I prefer to use a nested loop structure for clarity
# A=np.zeros((nk,num_occ_bands,ntf),dtype=complex)
# for k in range(nk):
#   # convert wfs from cell-periodic to true Bloch 
#   wfs_bloch=wfs[k,:,:]         # [nband,norb]
#   phases=get_phases_at_k(model,k_vec[k])
#   for i in range(norb):
#     wfs_bloch[:,i]=np.multiply(phases[:,i],wfs_bloch[:,i])
#   for n in range(num_occupied_bands):
#     for j in range(num_tf):
#       A[k,n,j]=np.vdot(wfs_bloch[j,:],tfs[j,:])
#       # I think vdot includes the complex conjugation of the first vector

# for k in range(nk):
#     wfs.solve_on_one_point([k_vec[k,0], k_vec[k,1]], [k])
#     # constructing A
#     A[k, :, :] = [[wfs[k][i][j].conj() for j in trial_orbitals] for i in range(num_occ_bands)]
#     # SVD 
#     V[k, :, :], S[k, :], W[k, :, :] = np.linalg.svd(A[k])


# def get_pos_ops(w0, supercell, orbs):
#     n_wf = w0.shape[2]
#     D = n_wf*len(supercell)
#     X = np.zeros((D, D), dtype=complex)
#     Y = np.zeros((D, D), dtype=complex)

#     idx1 = 0

#     for n in range(n_wf): # "band" index
#         for tx, ty in supercell: # cells in supercell
#             idx2 = 0
#             for m in range(n_wf): # "band" index
#                 for dx, dy in supercell: # cells in supercell 
#                     for i, orb in enumerate(orbs): # values of Wannier function on lattice
#                         w0n_r = w0[tx, ty, n, i] # Wannier function
#                         wRm_r = w0[dx, dy, m, i] # Wannier function
#                         pos = (orb[0] + tx)*lat_vecs[0] + (orb[1] + ty)*lat_vecs[1] # position
#                         X[idx1, idx2] += pos[0] * w0n_r * wRm_r.conj()
#                         Y[idx1, idx2] += pos[1] * w0n_r * wRm_r.conj() 
#                     idx2 += 1
#             idx1 += 1
            
#     return X, Y

# def get_omega_tilde(X, Y):
#     Omega_t = 0
#     for n in range(X.shape[0]):
#         for m in range(X.shape[1]):
#             if n != m:
#                 Omega_t += abs(X[n,m])**2

#     for n in range(Y.shape[0]):
#         for m in range(Y.shape[1]):
#             if n != m:
#                 Omega_t += abs(Y[n,m])**2

#     return Omega_t


def spread_real(w0, orbs, lat_vecs, decomp=False):
    """
    Spread functional computed in real space with Wannier functions

    Args:
        w0 (np.array): Wannier functions
        supercell (np.array): lattice translation vectors in reduced units
        orbs (np.array): orbital vectors in reduced units
        decomp (boolean): whether to separate gauge (in)variant parts of spread

    Returns:
        Omega: the spread functional
        Omega_inv: (optional) the gauge invariant part of the spread
        Omega_tilde: (optional) the gauge dependent part of the spread
        expc_rsq: \sum_{n} <r^2>_{n}
        expc_r_sq: \sum_n <\vec{r}>_{n}^2
    """
    nx, ny, n_occ = w0.shape[0], w0.shape[1], w0.shape[2]
    supercell = [(i,j) for i in range(-int((nx-nx%2)/2), int((nx-nx%2)/2)) 
                for j in range(-int((ny-ny%2)/2), int((ny-ny%2)/2))]
    r_n = np.zeros((n_occ, 2), dtype=complex) # <\vec{r}>_n
    rsq_n = np.zeros(n_occ, dtype=complex) # <r^2>_n
    R_nm = np.zeros((2, n_occ, n_occ, nx*ny), dtype=complex)

    expc_rsq = 0 # <r^2> 
    expc_r_sq = 0 # <\vec{r}>^2

    for n in range(n_occ): # "band" index
        for tx, ty in supercell: # cells in supercell
            for i, orb in enumerate(orbs): # values of Wannier function on lattice
                w0n_r = w0[tx, ty, n, i] # Wannier function
                pos = (orb[0] + tx)*lat_vecs[0] + (orb[1] + ty)*lat_vecs[1] # position
                r = np.sqrt(pos[0]**2 + pos[1]**2)

                # expectation value of position (vector)
                r_n[n, :] += np.array([pos[0] * abs(w0n_r)**2, pos[1] * abs(w0n_r)**2])
                rsq_n[n] += r**2 * w0n_r*w0n_r.conj()

                if decomp:
                    for m in range(n_occ):
                        for j, [dx, dy] in enumerate(supercell):
                            wRm_r = w0[(tx+dx)%nx, (ty+dy)%ny, n, i] # Wannier function
                            R_nm[0, n, m, j] += pos[0] * w0n_r*wRm_r.conj()
                            R_nm[1, n, m, j] += pos[0] * w0n_r*wRm_r.conj()

        expc_rsq += rsq_n[n]
        expc_r_sq += np.vdot(r_n[n, :], r_n[n, :])

    spread = expc_rsq - expc_r_sq

    if decomp:
        Omega_inv = 0 # gauge invariant part of spread
        Omega_tilde = 0 # gauge dependent part of spread

        Omega_inv += expc_rsq
        for i in range(R_nm.shape[1]):
            for j in range(R_nm.shape[2]):
                for k in range(R_nm.shape[3]):
                    Omega_inv -= np.vdot(R_nm[:, i, j, k], R_nm[:, i, j, k])
                    if not (i == j and supercell[k] == (0,0)):
                        Omega_tilde += np.vdot(R_nm[:, i, j, k], R_nm[:, i, j, k])

        assert np.allclose(spread, Omega_inv+Omega_tilde)
        return [spread, Omega_inv, Omega_tilde], expc_rsq, expc_r_sq

    else:
        return spread, expc_rsq, expc_r_sq

def get_projector(states, bands, Wannier=False):
    nx, ny = states.shape[0], states.shape[1]
    P_s = np.array([[ 
        np.sum( [np.outer(states[x, y, n, :].T, states[x, y, n, :].conj()) for n in bands], axis=0)
               for y in range(ny)] for x in range(nx)]) 
    
    P = (nx*ny if Wannier else 1) * np.sum(P_s, axis=(0,1))
    Q = np.eye(P.shape[0]) - P
    return P, Q


def k_overlap_mat(u_wfs, num_NN, orbs=None):
    """ 
    Compute the overlap matrix of Bloch eigenstates. Assumes that the last u_wf
    along each periodic direction corresponds to the next to last k-point in the 
    mesh (excludes endpoints). This way, the periodic boundary conditions are handled 
    internally.

    Args:
        u_wfs (np.array | wf_array): The cell periodic Bloch wavefunctions
        num_NN (int): number of nearest neighbors at a reciprocal lattice site
        n_occ (int): number of occupied eigenstates

    Returns:
        M (np.array): overlap matrix
    """
    if isinstance(u_wfs, wf_array):
        shape = u_wfs._wfs.shape
        orbs = u_wfs._model.get_orb()
    else:
        shape = u_wfs.shape
        assert orbs is not None, "Need to specify orbital vectors"

    nks = [*shape[0:2]]
    nkx, nky = nks[0], nks[1]
    n_states = shape[len(nks)]
    
    # assumes that there is no last element in the k mesh, so we need to introduce phases
    M = np.zeros((nkx, nky, num_NN, n_states, n_states), dtype=complex) # overlap matrix
    for n in range(n_states): # band index right (occupied)
        for m in range(n_states): # band index left (occupied)
            for kx in range(nkx):
                for ky in range(nky): 
                    for nn in range(num_NN): # nearest neighbors
                        # pbc at edge of BZ
                        if nn == 0:
                            M[kx, ky, nn, m, n] = np.vdot(
                                u_wfs[kx, ky][m, :], u_wfs[(kx+1)%nkx, ky][n, :] *
                                (np.array(np.exp(-1j*2*np.pi*orbs @ np.array([1,0]).T ), dtype=complex).T if kx == nkx-1
                                 else 1) 
                                )
                        elif nn == 1:
                            M[kx, ky, nn, m, n] = np.vdot(
                                u_wfs[kx, ky][m, :], u_wfs[kx, (ky+1)%nky][n, :] *
                                (np.array(np.exp(-1j*2*np.pi*orbs @ np.array([0,1]).T ), dtype=complex).T if ky == nky-1
                                 else 1) 
                                )
                        elif nn == 2:
                            M[kx, ky, nn, m, n] = np.vdot(
                                u_wfs[kx, ky][m, :], u_wfs[kx-1, ky][n, :] *
                                (np.array(np.exp(-1j*2*np.pi*orbs @ np.array([-1,0]).T ), dtype=complex).T if kx == 0
                                 else 1) 
                                )
                        elif nn == 3:
                            M[kx, ky, nn, m, n] = np.vdot(
                                u_wfs[kx, ky][m, :], u_wfs[kx, ky-1][n, :] *
                                (np.array(np.exp(-1j*2*np.pi*orbs @ np.array([0,-1]).T ), dtype=complex).T if ky == 0
                                 else 1) 
                                 )
    return M
    
    # print(np.allclose(M[nkx-1, :, 0, 0, 0], M[0, :, 2, 0, 0].conj()))       

    # M2 = np.zeros((nkx, nky, num_NN, n_occ, n_occ), dtype=complex) # overlap matrix
    # for n in range(n_occ): # band index right (occupied)
    #     for m in range(n_occ): # band index left (occupied)
    #          # neighbors right and left
    #          for ky in range(nky):
    #              for kx in range(nkx-1):
    #                  M_xp = np.vdot(u_wfs[kx, ky][m, :], u_wfs[kx+1,ky][n, :])
    #                  M2[kx, ky, 0, m, n] = M_xp
    #                  M2[kx+1, ky, 2, n, m] = M_xp.conj()
    #              M_xpb = np.vdot(u_wfs[nkx-1, ky][m, :], u_wfs[0, ky][n,:])
    #              M2[nkx-1, ky, 0, m, n] = M_xpb
    #              M2[0, ky, 2, n, m] = M_xpb.conj()
    #          # up and down
    #          for kx in range(nkx):
    #              for ky in range(nky-1):
    #                  M_yp  = np.vdot(u_wfs[kx, ky][m, :], u_wfs[kx,ky+1][n, :])
    #                  M2[kx, ky, 1, m, n] = M_yp
    #                  M2[kx, ky+1, 3, n, m] = M_yp.conj()
    #              M_ypb = np.vdot(u_wfs[kx, nky-1][m, :], u_wfs[kx, 0][n,:])
    #              M2[kx, nky-1, 1, m, n] = M_ypb
    #              M2[kx, 0, 3, n, m] = M_ypb.conj()

    # # Assumes a full BZ s.t. the first and last k point are equivalent up to phase for cell periodic.
    # M = np.zeros((nkx-1, nky-1, num_NN, n_occ, n_occ), dtype=complex) # overlap matrix
    # for n in range(n_occ): # band index right (occupied)
    #     for m in range(n_occ): # band index left (occupied)
    #         for kx in range(nkx-1):
    #             for ky in range(nky-1): 
    #                 for nn in range(num_NN): # nearest neighbors
    #                     # pbc at edge of BZ
    #                     if nn == 0:
    #                         M[kx, ky, nn, m, n] = np.vdot(u_wfs[kx, ky][m, :], u_wfs[kx+1, ky][n, :])
    #                     elif nn == 1:
    #                         M[kx, ky, nn, m, n] = np.vdot(u_wfs[kx, ky][m, :], u_wfs[kx, ky+1][n, :])
    #                     elif nn == 2:
    #                         M[kx, ky, nn, m, n] = np.vdot(u_wfs[kx, ky][m, :], u_wfs[kx-1, ky][n, :])
    #                     elif nn == 3:
    #                         M[kx, ky, nn, m, n] = np.vdot(u_wfs[kx, ky][m, :], u_wfs[kx, ky-1][n, :])

    # print(np.allclose(M, M2))
    
    return M

def spread_recip(M, w_b, b_vec, decomp=False):
    """
    Args:
        M (np.array): 
            overlap matrix
        w_b (int): 
            finite difference weights
        b_vec (list[np.array]): 
            vectors connected NN reciprocal lattice sites
        decomp (bool, optional): 
            Whether to compute and return decomposed spread. Defaults to False.

    Returns:
        spread | [spread, Omega_i, Omega_tilde], expc_rsq, expc_r_sq : 
            quadratic spread, the expectation of the position squared,
            and the expectation of the position vector squared
    """
    nkx, nky = M.shape[0], M.shape[1] # mesh size
    N_k = nkx*nky # number of k points in first BZ
    r_n = np.zeros((n_occ, 2), dtype=complex) # <\vec{r}>_n
    rsq_n = np.zeros(n_occ, dtype=complex) # <r^2>_n
    expc_rsq = 0 # <r^2> 
    expc_r_sq = 0 # <\vec{r}>^2

    for n in range(n_occ):
        for kx in range(nkx):
            for ky in range(nky):
                for idx, b in enumerate(b_vec):
                    r_n[n, :] += -(1/N_k) * w_b * b * np.log(M[kx, ky, idx, n, n]).imag
                    rsq_n[n] += (1/N_k) * w_b * (1 - abs(M[kx, ky, idx, n, n])**2 + np.log(M[kx, ky, idx, n, n]).imag**2)
                    
        expc_rsq += rsq_n[n] # <r^2> 
        expc_r_sq += np.vdot(r_n[n, :], r_n[n, :]) # <\vec{r}>^2
        
    spread = expc_rsq - expc_r_sq
                                
    if decomp:
        Omega_i = 0
        Omega_tilde = 0
        for kx in range(nkx):
            for ky in range(nky):
                for idx, b in enumerate(b_vec):
                    Omega_i += (1/N_k) * w_b * n_occ 
                    for n in range(n_occ):
                        Omega_tilde += (1/N_k) * w_b * (-np.log(M[kx, ky, idx, n, n]).imag - np.vdot(b, r_n[n]))**2
                        for m in range(n_occ):
                            Omega_i -= (1/N_k) * w_b * abs(M[kx, ky, idx, m, n])**2
                            if m != n:
                                Omega_tilde += (1/N_k) * w_b * abs(M[kx, ky, idx, m, n])**2
                                
        return [spread, Omega_i, Omega_tilde], expc_rsq, expc_r_sq
    
    else: 
        return spread, expc_rsq, expc_r_sq
    
def get_pbc_phase(orbs, G):
    """
    Get phase factors for cell periodic pbc across BZ boundary

    Args:
        orbs (np.array): reduced coordinates of orbital positions
        G (list): reduced coordinates of reciprocal lattice vector

    Returns:
        phase: phase factor to be multiplied to last cell periodic eigenstates
        in k-mesh 
    """
    phase = np.array(np.exp(-1j*2*np.pi*orbs @ np.array(G).T ), dtype=complex).T
    return phase