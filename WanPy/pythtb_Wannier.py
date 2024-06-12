import numpy as np
from pythtb import *
from itertools import product
from itertools import combinations_with_replacement as comb
from scipy.linalg import expm

def get_recip_lat_vecs(model):
    lat_vecs = model.get_lat()
    b = 2 * np.pi * np.linalg.inv(lat_vecs).T
    return b

# TODO: figure out factor of 2 
def get_weights(*nks, model, N_sh=1):
    b_vec = get_recip_lat_vecs(model)
    dk = np.array([b_vec[i]/nk for i, nk in enumerate(nks)])
    dim = b_vec.shape[0]
    Cart_idx = list(comb(range(dim), 2))
    n_comb = len(Cart_idx)

    A = np.zeros((n_comb, N_sh))
    q = np.zeros((n_comb))

    for s in range(N_sh):
        for j, (alpha, beta) in enumerate(Cart_idx): 
            if alpha == beta:
                q[j] = 1
            for k in dk:
                A[j, s] += k[alpha] * k[beta]
   
    U, D, Vt = np.linalg.svd(A, full_matrices=False)
    w = (Vt.T @ np.linalg.inv(np.diag(D)) @ U.T) @ q
    return w/2

def gen_k_mesh(*nks, centered=False, flat=True, endpoint=False):
    """Generate k-mesh in reduced coordinates

    Args:
        nks (tuple(int)): tuple of number of k-points along each reciprocal lattice basis vector
        centered (bool, optional): Whether Gamma is at origin or ceneter of mesh. Defaults to False.
        flat (bool, optional): 
          If True returns rank 1 matrix of k-points,
          If False returns rank 2 matrix of k-points. Defaults to True.
        endpoint (bool, optional): If True includes both borders of BZ. Defaults to False.

    Returns:
        k-mesh (np.array): list of k-mesh coordinates  
    """
    end_pts = [-0.5, 0.5] if centered else [0,1]

    k_vals = [np.linspace(end_pts[0], end_pts[1], nk, endpoint=endpoint) for nk in nks]
    mesh = np.array(list(product(*k_vals)))

    return mesh if flat else mesh.reshape(*[nk for nk in nks], len(nks))

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
  lam = -1 if inverse else 1  # overall minus if getting cell periodic from Bloch
  orb = model.get_orb()   # numpy array in order [orbital, reduced coord]
  per_dir = model._per    # list of periodic dimensions
  # slice second dimension to only keep only periodic dimensions in orb
  per_orb = orb[:, per_dir]
 
  # compute a list of phase factors [k_val, orbital]
  wf_phases = np.exp(lam * 1j* 2 * np.pi * per_orb @ k_vec.T, dtype=complex).T
  return wf_phases  # 1D numpy array of dimension norb

def get_bloch_wfs(model, u_wfs, k_mesh, inverse=False):
    """
    Change the cell periodic wfs to Bloch wfs

    Args:
    model (pythtb.tb_model): PythTB model
    wfs (pythtb.wf_array): cell periodic wfs [k, nband, norb]
    k_mesh (np.array): k-mesh on which u_wfs is defined
        
    Returns:
    wfs_psi: np.array
        wfs with orbitals multiplied by proper phase factor

    """
    if isinstance(u_wfs, wf_array):
        shape = u_wfs._wfs.shape # [*nks, idx, orb]
        u_wfs = u_wfs._wfs
    else:
        shape = u_wfs.shape # [*nks, idx, orb]

    nks = shape[:-2]
    norb = model.get_num_orbitals() # number of orbitals

    if len(k_mesh.shape) > 2:
        k_mesh = k_mesh.reshape(np.prod(nks), len(nks)) # flatten

    # Phases come in a list flattened over k space
    # Needs be reshaped to match k indexing of wfs
    phases = get_orb_phases(model, k_mesh, inverse=inverse).reshape(*nks, norb)
    psi_wfs = u_wfs * phases[..., np.newaxis, :]  # Broadcasting the phases to match dimensions 

    return psi_wfs

def set_trial_function(tf_list, norb):
  """
  Args:
      tf_list: list[int | list[tuple]]
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
    if isinstance(tf, (int, np.int64)): 
      # We only have a trial function on one site
      tfs[j, tf] = 1
    elif isinstance(tf, (list, np.ndarray)):
      # Must be list of tuples of the form (site, amplitude)
      for site, amp in tf:
        tfs[j, site] = amp
      # normalizing 
      tfs[j,:] /= np.sqrt(sum(abs(tfs[j, :])))
    else:
      raise TypeError("tf_list is not of apporpriate type")

  # return numpy array containing trial functions
  return tfs # tfs in order[trial funcs, orbitals]

def tf_overlap_mat(
        psi_wfs, tfs, state_idx
        ):
    """

    Args:
        psi_wfs (np.array): Bloch eigenstates
        tfs (np.array): trial wfs
        state_idx (list): band indices to form overlap matrix with
        switch_rep (bool, optional): For testing. Defaults to False.
        tfs_swap (np.array, optional): For testing. Defaults to None.

    Returns:
        A (np.array): overlap matrix
    """
    nks = psi_wfs.shape[:-2]
    ntfs = tfs.shape[0]

    A = np.zeros((*nks, len(state_idx), ntfs), dtype=complex)
    for n in state_idx:
        for j in range(ntfs):
            A[..., n, j] = psi_wfs.conj()[..., n, :] @ tfs[j, :]

    return A

def get_psi_tilde(
        psi_wf, tf_list, state_idx = None,
        compact_SVD=False
        ):

    shape = psi_wf.shape
    n_orb = shape[-1]
    n_state = shape[-2]
    nks = shape[:-2]
    n_occ = int(n_state/2) # assuming half filled

    if state_idx is None: # assume we are Wannierizing occupied bands at half-filling 
        state_idx = list(range(0, n_occ)) 

    tfs = set_trial_function(tf_list, n_orb)
    A = tf_overlap_mat(psi_wf, tfs, state_idx)
    V, _, Wh = SVD(A, full_matrices=False, compact_SVD=compact_SVD)

    # swap only last two indices in transpose (ignore k indices)
    # slice psi_wf to keep only occupied bands
    psi_tilde = (
        (V @ Wh).transpose(*([i for i in range(len(nks))] +  [len(nks)+1, len(nks)])) @ psi_wf[..., state_idx, :]
     ) # [*nk_i, nband, norb]
    return psi_tilde

def SVD(A, full_matrices=False, compact_SVD=False):
    # SVD on last 2 axes by default (preserving k indices)
    V, S, Wh = np.linalg.svd(A, full_matrices=full_matrices)

    if compact_SVD: # testing
        assert A.shape[2:][0] ==  A.shape[2:][1] # square
        V, S, Wh = np.linalg.svd(A, full_matrices=True)
        V = V[..., :, :-1]
        S = S[..., :-1]
        Wh = Wh[..., :-1, :]
    
    return V, S, Wh

def DFT(psi_tilde, norm=None):
    dim_k = len(psi_tilde.shape[:-2])
    Rn = np.fft.ifftn(psi_tilde, axes = [i for i in range(dim_k)], norm=norm)
    return Rn

def Wannierize(
        model, u_wfs, tf_list, state_idx = None,
        k_mesh=None, compact_SVD=False, ret_psi_til=False
        ):
    """
    Obtains Wannier functions cenetered in home unit cell.

    Args:
        model (pythtb.model): PythTB tight-binding model 
        u_wfs (pythtb.wf_array): wf array on k-mesh. k-mesh must not be full.
        tf_list (list): list of sites and amplitudes of trial wfs
        n_occ (int): number of occupied states to Wannierize from

        compact_SVD (bool, optional): For testing purposes. Defaults to False.
        switch_rep (bool, optional): For testing purposes. Defaults to False.
        tfs_swap (list, optional): For testing purposes. Defaults to None.
        
    Returns:
        w_0n (np.array): Wannier functions in home unit cell
    """
    if isinstance(u_wfs, wf_array):
        shape = u_wfs._wfs.shape # [*nks, idx, orb]
    else:
        shape = u_wfs.shape # [*nks, idx, orb]

    nks = shape[:-2]

    # get Bloch wfs
    if k_mesh is None: # assume u_wfs is defined over full BZ
        k_mesh = gen_k_mesh(*nks, flat=True, endpoint=False)
    psi_wfs = get_bloch_wfs(model, u_wfs, k_mesh)
    # get tilde states
    psi_tilde = get_psi_tilde(psi_wfs, tf_list, state_idx=state_idx, compact_SVD=compact_SVD)

    # get Wannier functions
    w_0n = DFT(psi_tilde)
    
    if ret_psi_til:
        return w_0n, psi_tilde
    return w_0n

#### Spread functions ######

#TODO: Allow for arbitrary dimensions
def spread_real(model, w0, decomp=False):
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
        expc_rsq: \sum_n <r^2>_{n}
        expc_r_sq: \sum_n <\vec{r}>_{n}^2
    """
    lat_vecs = model.get_lat() # lattice vectors
    orbs = model.get_orb()
    # shape = w0.shape # [*nks, idx, orb]

    # nxs = shape[:-2]
    # n_orb = shape[-1]
    # n_states = shape[-2]

    # assuming 2D for now
    nx, ny, n_occ = w0.shape[0], w0.shape[1], w0.shape[2]
    # translation vectors in reduced units
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
                            wRm_r = w0[(tx+dx)%nx, (ty+dy)%ny, m, i] # translated Wannier function
                            R_nm[0, n, m, j] += pos[0] * w0n_r*wRm_r.conj()
                            R_nm[1, n, m, j] += pos[1] * w0n_r*wRm_r.conj()

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
        return [spread, Omega_inv, Omega_tilde], r_n, rsq_n

    else:
        return spread, r_n, rsq_n


def k_overlap_mat(u_wfs, orbs=None):
    """ 
    Compute the overlap matrix of Bloch eigenstates. Assumes that the last u_wf
    along each periodic direction corresponds to the next to last k-point in the 
    mesh (excludes endpoints). This way, the periodic boundary conditions are handled 
    internally.

    Args:
        u_wfs (np.array | wf_array): The cell periodic Bloch wavefunctions

    Returns:
        M (np.array): overlap matrix
    """
    if isinstance(u_wfs, wf_array):
        shape = u_wfs._wfs.shape # [*nks, idx, orb]
        orbs = u_wfs._model.get_orb()
        u_wfs = np.array(u_wfs._wfs)
    else:
        shape = u_wfs.shape # [*nks, idx, orb]
        # need access to orbital vectors for pbc
        assert orbs is not None, "Need to specify orbital vectors"
        
    nks = shape[:-2]
    n_states = shape[-2]

    # Below works for arbitrary dimension but is more convoluted
    # nnbr_mask = list(product([-1, 0, 1], repeat=len(nks)))
    # b_idx_arr = np.array([nnbr_vec for nnbr_vec in nnbr_mask if sum(abs(nnbr) for nnbr in nnbr_vec) == 1], dtype=int) # index translation array

    b_idx_arr = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])] # vector connecting nearest neighboring k indices
    
    k_rng = [range(nk) for nk in nks] # list of dim(dim_k) with all k indices along each dir
    k_idx_arr = list(product(*k_rng)) # all pairwise combinations of k_indices
    
    # assumes that there is no last element in the k mesh, so we need to introduce phases
    M = np.zeros((*nks, len(b_idx_arr), n_states, n_states), dtype=complex) # overlap matrix
    for k in k_idx_arr:
        for idx, b in enumerate(b_idx_arr): # nearest neighbors
            k_nbr = np.array(k) + b
            cross_bnry = True if np.any(np.in1d(k_nbr, [-1, *nks])) else False
            bc_phase = np.array(np.exp(-1j*2*np.pi*orbs @ b.T ), dtype=complex).T if cross_bnry else 1
            # apply pbc
            k_nbr = np.mod(k_nbr, nks)

            for n in range(n_states): # band index right (occupied)
                for m in range(n_states): # band index left (occupied)
                    M[k][idx, m, n] = np.vdot(u_wfs[k][m, :], u_wfs[tuple(k_nbr)][n, :] * bc_phase )
    return M

def spread_recip(model, M, decomp=False):
    """
    Args:
        M (np.array): 
            overlap matrix
        decomp (bool, optional): 
            Whether to compute and return decomposed spread. Defaults to False.

    Returns:
        spread | [spread, Omega_i, Omega_tilde], expc_rsq, expc_r_sq : 
            quadratic spread, the expectation of the position squared,
            and the expectation of the position vector squared
    """
    shape = M.shape
    nks = M.shape[:-3]
    Nk = np.prod(nks)
    n_states = shape[3]

    recip_lat_vecs = get_recip_lat_vecs(model)
    dk = [recip_lat_vecs[i]/nk for i, nk in enumerate(nks)]

    w_b = get_weights(nks, model=model)[0] # Assumes only one shell (and factor of 2 error somehow)

    k_vals = [range(nk) for nk in nks] # list of dim(dim_k) with all k indices along each dir
    k_idx_arr = list(product(*k_vals)) # all pairwise combinations of k_indices

    # nnbr_mask = list(product([-1, 0, 1], repeat=len(nks)))
    # b_idx_arr = np.array([nnbr_vec for nnbr_vec in nnbr_mask if sum(abs(nnbr) for nnbr in nnbr_vec) == 1], dtype=int) # index translation array
    b_idx_arr = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])] # vectors connecting nearest neighbor k points
    
    r_n = np.zeros((n_states, 2), dtype=complex) # <\vec{r}>_n
    rsq_n = np.zeros(n_states, dtype=complex) # <r^2>_n
    expc_rsq = 0 # <r^2> 
    expc_r_sq = 0 # <\vec{r}>^2

    for n in range(n_states):
        for k in k_idx_arr:
            for idx, b_idx in enumerate(b_idx_arr):
                b = b_idx @ dk
                r_n[n, :] += -(1/Nk) * w_b * b * np.log(M[k][idx, n, n]).imag
                rsq_n[n] += (1/Nk) * w_b * (1 - abs(M[k][idx, n, n])**2 + np.log(M[k][idx, n, n]).imag**2)
                    
        expc_rsq += rsq_n[n] # <r^2> 
        expc_r_sq += np.vdot(r_n[n, :], r_n[n, :]) # <\vec{r}>^2
        
    spread = expc_rsq - expc_r_sq
                                
    if decomp:
        Omega_i = 0
        Omega_tilde = 0
        for k in k_idx_arr:
            for idx, b_idx in enumerate(b_idx_arr):
                b = b_idx @ dk
                Omega_i += (1/Nk) * w_b * n_states 
                for n in range(n_states):
                    Omega_tilde += (1/Nk) * w_b * (-np.log(M[k][idx, n, n]).imag - np.vdot(b, r_n[n]))**2
                    for m in range(n_states):
                        Omega_i -= (1/Nk) * w_b * abs(M[k][idx, m, n])**2
                        if m != n:
                            Omega_tilde += (1/Nk) * w_b * abs(M[k][idx, m, n])**2
                                
        return [spread, Omega_i, Omega_tilde], r_n, rsq_n
    
    else: 
        return spread, r_n, rsq_n

###### helper functions #####
def get_pbc_phase(orbs, G):
    """
    Get phase factors for cell periodic pbc across BZ boundary

    Args:
        orbs (np.array): reduced coordinates of orbital positions
        G (list): reciprocal lattice vector in reduced coordinates 

    Returns:
        phase: phase factor to be multiplied to last cell periodic eigenstates
        in k-mesh 
    """
    phase = np.array(np.exp(-1j*2*np.pi*orbs @ np.array(G).T ), dtype=complex).T
    return phase

def swap_reps(eigvecs, k_points, swap_pts, swap_scheme):
    swap_eigvecs = eigvecs.copy()
    nks = eigvecs.shape[:-2]

    diff = np.linalg.norm(k_points-np.array(swap_pts), axis=len(nks))
    high_sym_idx = np.where(diff == np.min(diff))

    if len(nks) == 1:
        for k in zip(*high_sym_idx):
            for src_bnd, targ_bnd in swap_scheme.items():
                swap_eigvecs[k, src_bnd, :] = eigvecs[k, targ_bnd, :]
                swap_eigvecs[k, targ_bnd, :] = eigvecs[k, src_bnd, :]

    if len(nks) == 2:
        for kx, ky in zip(*high_sym_idx):
            for src_bnd, targ_bnd in swap_scheme.items():
                swap_eigvecs[kx, ky, src_bnd, :] = eigvecs[kx, ky, targ_bnd, :]
                swap_eigvecs[kx, ky, targ_bnd, :] = eigvecs[kx, ky, src_bnd, :]

    return swap_eigvecs

####### Wannier interpolation ########

def diag_h_in_subspace(model, eigvecs, k_path, ret_evecs=False):
    """
    Diagonalize the Hamiltonian in a projected subspace

    Args:
        model (pythtb.model): 
            model to obtain Bloch Hamiltonian
        eigvecs (np.array | pythtb.wf_array): 
            Eigenvectors spanning the target subspace
        k_path (np.array): 
            1D path on which we want to diagonalize the Hamiltonian

    Returns:
        eigvals (np.array):
            eigenvalues in subspace
    """
    if isinstance(eigvecs, wf_array):
        shape = eigvecs._wfs.shape # [*nks, idx, orb]
    else:
        shape = eigvecs.shape # [*nks, idx, orb]

    nks = shape[:-2]
    n_orb = shape[-1]
    n_states = shape[-2]

    H_k_proj = np.zeros([*nks, n_states, n_states], dtype=complex)

    for k_idx, k in enumerate(k_path):
        H_k = model._gen_ham(k)
        V = np.transpose(eigvecs[k_idx], axes=(1,0)) # [orb, num_evecs]
        H_k_proj[k_idx, :, :] = V.conj().T @ H_k @ V # projected Hamiltonian

    eigvals = np.zeros((*nks, n_states), dtype=complex)
    evecs = np.zeros((*nks, n_states, n_orb), dtype=complex)

    for idx, k in enumerate(k_path):
        eigvals[idx, :], evec = np.linalg.eigh(H_k_proj[idx]) # [k, n], [evec wt, n]
        for i in range(evec.shape[1]): 
            # Returns in given eigvec basis
            evecs[idx, i, :] = sum([evec[j, i] * eigvecs[idx, j, :] for j in range(evec.shape[0])])

    if ret_evecs:
        return eigvals.real, evecs
    else:
        return eigvals.real
    

####### Maximally Localized WF ############

def find_optimal_subspace(
        model, outer_states, inner_states, tol = 2e-4, full_mesh=False, k_mesh=None, iter_num=100,
        print_=False):
    """
    Assumes the states are defined in reciprocal space and are 
    of Bloch character (cell periodic), and that we have a square
    lattice (finite difference weights are only specified for cubic lattice).
    This also assumes there is a 1-1 correspondence with the state indices and 
    the k-mesh that they are defined on. E.g. index [0,0] on a 2D-mesh corresponds
    to the Gamma point. This is so periodic boundary conditions can be applied with
    taking the reduced reciprocal lattice vector to be the index translation e.g. [0, 1]
    when at the [., nky] index, or [-1, 0] when at the [0, .] index. If there is not a 1-1 
    correspondence, then we have to specify the k-mesh that matches with where the states
    were defined.  

    Args:
        outer_states: States spanning outer space 
        inner_states: States spanning a subspace of outer space
        full_mesh: If True, then we assume the eigenstates have already had pbc applied to them

    Returns:
        states_min: States spanning optimal subspace
    """
    if isinstance(inner_states, wf_array):
        shape = inner_states._wfs.shape # [*nks, idx, orb]
        inner_states = np.array(inner_states._wfs)
    else:
        shape = inner_states.shape # [*nks, idx, orb]

    nks = shape[:-2]
    Nk = np.prod(nks)
    n_orb = shape[-1] 
    n_states = shape[-2]
    dim_subspace = n_states

    lat_vecs = model.get_lat() # lattice vectors
    orbs = model.get_orb()
    per_dir = model._per    # list of periodic dimensions
    # slice second dimension to only keep only periodic dimensions in orb
    per_orbs = orbs[:, per_dir]

    k_vals = [range(nk) for nk in nks] # list of dim = dim_k with all k indices along each dir
    k_idx_arr = list(product(*k_vals)) # all pairwise combinations of k_indices

    # 2D k-mesh
    num_NN = 2*len(nks) # number of nearest neighbors to a given k point where states defined
    G = 2*np.pi/lat_vecs[0][0] # reciprocal lattice vector
    b = G / nks[0]  # spacing between allowed kx
    w_b = len(nks)/(num_NN * b**len(nks)) # finite difference weights for cubic lattice
    nnbr_mask = list(product([-1, 0, 1], repeat=len(nks)))
    b_idx = np.array([nnbr_vec for nnbr_vec in nnbr_mask if sum(abs(nnbr) for nnbr in nnbr_vec) == 1], dtype=int) # index translation array 

    # Projector on initial subspace at each k
    P = np.zeros((*nks, n_orb, n_orb), dtype=complex)
    Q = np.zeros((*nks, n_orb, n_orb), dtype=complex)
    for k in k_idx_arr:
        P[k][:, :] = np.sum([np.outer(inner_states[k][n, :], inner_states[k][n, :].conj() ) for n in range(int(n_states))], axis=0)
        Q[k][:, :] = np.eye(P[k].shape[0]) - P[k]

    # Projector on initial subspace at each k (for pbc of neighboring spaces)
    P_nbr = np.zeros((*nks, num_NN, n_orb, n_orb), dtype=complex)
    Q_nbr = np.zeros((*nks, num_NN, n_orb, n_orb), dtype=complex)
    T_kb = np.zeros((*nks, num_NN), dtype=complex)

    for k in k_idx_arr:
        for idx, t_k in enumerate(b_idx):
            k_nbr_idx = tuple(np.array(k) + t_k)
            k_nbr_idx_pbc = tuple(np.mod(k_nbr_idx, nks))

            # TODO: This needs fixed. I don't think it correctly identifies boundaries since
            # the difference between neighboring k-points will never be 1 on a non-full k-mesh
            # but we could've crossed boundary (e.g. [0, 0.9...] -> [0, 0]) 
            if k_mesh is not None: # first and last index not necessarily at border
                k_pt = k_mesh[k]
                knbr_pt = k_mesh[k_nbr_idx_pbc]
                G = knbr_pt - k_pt # reciprocal lattice vector separating neighbors (dimensionless)
                cross_bndry =  np.any(np.in1d(G, [1, -1]))
            else:
                # if we were at 0th or last index
                cross_bndry = np.any(np.in1d(k_nbr_idx, [-1, *nks]))
                G = t_k
            # if full_mesh then phases already included at boundary
            if cross_bndry and not full_mesh:
                # assumes first and last indices are at edge of BZ
                bc_phase = np.exp(-1j*2*np.pi* per_orbs @ G.T, dtype=complex).T 
            else:
                bc_phase = 1

            # apply pbc
            state_pbc = inner_states[k_nbr_idx_pbc] * bc_phase
            P_nbr[k][idx, :, :] = np.sum([np.outer(state_pbc[n].T, state_pbc[n].conj()) for n in range(int(n_states))], axis=0)
            Q_nbr[k][idx, :, :] = np.eye(n_orb) - P_nbr[k][idx, :, :] 

    P_min = np.copy(P) # start of iteration
    P_nbr_min = np.copy(P_nbr) # start of iteration
    Q_nbr_min = np.copy(Q_nbr) # start of iteration

    # states spanning optimal subspace minimizing gauge invariant spread
    states_min = np.zeros((*nks, dim_subspace, n_orb), dtype=complex)

    # diff = None
    for i in range(iter_num):
        for k in k_idx_arr:
            P_avg = np.sum(w_b * P_nbr_min[k], axis=0)
            
            # diagonalizing P_avg in outer_states basis
            N = outer_states.shape[-2]
            Z = np.zeros((N, N), dtype=complex)
            for n in range(N):
                for m in range(N):
                    Z[m, n] = outer_states[k][m, :].conj() @ (P_avg @ outer_states[k][n, :])
            # Z = np.einsum('ni,nj->ij', outer_states[k].conj(), P_avg @ outer_states[k])

            eigvals, eigvecs = np.linalg.eigh(Z) # [val, idx]
            for idx, n in enumerate(np.argsort(eigvals.real)[-dim_subspace:]): # keep ntfs wfs with highest eigenvalue
                states_min[k][idx, :] = np.sum([eigvecs[i, n] * outer_states[k][i, :] for i in range(eigvecs.shape[0])], axis=0)

            P_new = np.einsum('ni,nj->ij', states_min[k], states_min[k].conj())
            # diff = np.linalg.norm(P_min[k] - P_new)
            alpha = 1 # mixing with previous step to break convergence loop
            P_min[k] = alpha * P_new + (1-alpha) * P_min[k] # for next iteration
            
            for idx, t_k in enumerate(b_idx):
                k_nbr_idx = tuple(np.array(k) + t_k)
                k_nbr_idx_pbc = tuple(np.mod(k_nbr_idx, nks))

                if k_mesh is not None:
                    k_pt = k_mesh[k]
                    knbr_pt = k_mesh[k_nbr_idx_pbc]
                    G = knbr_pt - k_pt
                    # if G is a reciprocal lattice vector we crossed boundary
                    cross_bndry =  np.any(np.in1d(G, [1, -1]))
                else: # we assume that the first and last indices in the k mesh are at the boundary
                    # if we were at 0th or last index
                    cross_bndry = np.any(np.in1d(k_nbr_idx, [-1, *nks]))
                    G = t_k

                if cross_bndry and not full_mesh:
                    # assumes first and last indices are at edge of BZ
                    bc_phase = np.array(np.exp(-1j*2*np.pi* per_orbs @ G.T), dtype=complex).T 
                else:
                    bc_phase = 1

                # apply pbc
                state_pbc = states_min[k_nbr_idx_pbc] * bc_phase
                P_nbr_min[k][idx, :, :] = np.einsum('ni,nj->ij', state_pbc, state_pbc.conj())
                Q_nbr_min[k][idx, :, :] = np.eye(n_orb) - P_nbr_min[k][idx, :, :] 
                T_kb[k][idx] = np.trace(P_min[k] @ Q_nbr_min[k][idx, :, :])

        if print_:
            Omega_I = (1/Nk) * w_b * np.sum(T_kb)
            print(f"{i} Omega_I: {Omega_I.real}")

    return states_min


def find_min_unitary(model, M, eps=1/160, iter_num=10, print_=False):

    shape = M.shape
    nks = shape[:-3]
    dim_k = len(nks)
    Nk = np.prod(nks)
    numNN = shape[-3]
    num_state = shape[-1]

    recip_lat_vecs = get_recip_lat_vecs(model) # reciprocal lattice vectors
    dk = np.array([recip_lat_vecs[i]/nk for i, nk in enumerate(nks)]) # basis vectors of spacing between allowed crystal momenta

    w_b = get_weights(nks, model=model)[0] # Assumes only one shell (and factor of 2 error somehow)

    k_vals = [range(nk) for nk in nks] # list of dim(dim_k) with all k indices along each dir
    k_idx_arr = list(product(*k_vals)) # all pairwise combinations of k_indices

    # nnbr_mask = list(product([-1, 0, 1], repeat=len(nks)))
    # b_idx_arr = np.array([nnbr_vec for nnbr_vec in nnbr_mask if sum(abs(nnbr) for nnbr in nnbr_vec) == 1], dtype=int) # index translation array
    b_idx_arr = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])] # vectors connecting nearest neighbor k points

    q = np.zeros((*nks, numNN, num_state), dtype=complex)
    R = np.zeros((*nks, numNN, num_state, num_state), dtype=complex) 
    T = np.zeros((*nks, numNN, num_state, num_state), dtype=complex)
    G = np.zeros((*nks, num_state, num_state), dtype=complex)
    r_n = np.zeros((num_state, dim_k), dtype=complex) # <\vec{r}>_n
    U = np.zeros(G.shape, dtype=complex) # unitary transformation
    dW = np.zeros(G.shape, dtype=complex) # anti-Hermitian matrix
    M0 = np.copy(M) # initial overlap matrix
    M = np.copy(M) # new overlap matrix

    # initializing
    spread, _, _ = spread_recip(model, M, decomp=True)
    omega_tilde_prev = spread[2]
    U[...] = np.eye(num_state, dtype=complex) # initialize as identity
    
    for i in range(iter_num):
        G.fill(0)
        r_n.fill(0)
        dW.fill(0)

        for n in range(num_state):
            r_n[n, :] = - (1/Nk) * w_b * np.sum(
                [b_idx @ dk * np.log(M[k][idx, n, n]).imag 
                for k in k_idx_arr for idx, b_idx in enumerate(b_idx_arr)], axis=0)
                
        for k in k_idx_arr:
            for idx, b_idx in enumerate(b_idx_arr):
                b = b_idx @ dk # reciprocal translation vec
                k_nbr_idx = tuple(np.array(k) + b_idx)
                k_nbr_idx = np.mod(k_nbr_idx, nks)
                for n in range(num_state):
                    q[k][idx, n] = np.log(M[k][idx, n, n]).imag + np.vdot(b, r_n[n])
                    R[k][idx, :, n] = M[k][idx, :, n] * M[k][idx, n, n].conj()
                    T[k][idx, :, n] = (M[k][idx, :, n] /  M[k][idx, n, n]) * q[k][idx, n]
                    
                A_R = (R[k][idx] - R[k][idx].conj().T) / 2
                S_T = (T[k][idx] + T[k][idx].conj().T) / (2j)
                G[k] += 4 * w_b * ( A_R - S_T )

            dW[k] = eps * G[k]
            U[k] = U[k] @ expm(dW[k])

        for k in k_idx_arr:
            for idx, b_idx in enumerate(b_idx_arr):
                k_nbr_idx = tuple(np.array(k) + b_idx)
                k_nbr_idx = np.mod(k_nbr_idx, nks)
                M[k][idx, :, :] = U[k].conj().T @ M0[k][idx, :, :] @ U[tuple(k_nbr_idx)]

        spread, _, _ = spread_recip(model, M, decomp=True)
        omega_tilde = spread[2]

        if omega_tilde > omega_tilde_prev:
            print("Warning: Omega_tilde increasing. Decreasing eps by 10%.")
            eps = eps * 0.9

        if print_:
            print(f"{i} Omega_til = {omega_tilde.real}, Grad mag: {np.linalg.norm(np.sum(G, axis=(0,1)))}")
        
        omega_tilde_prev = omega_tilde
        
    return U, M

def get_max_loc_uwfs(
        model, u_wfs, eps=1/160, iter_num=10, print_=False
        ):
    
    if isinstance(u_wfs, wf_array):
        shape = u_wfs._wfs.shape # [*nks, idx, orb]
    else:
        shape = u_wfs.shape # [*nks, idx, orb]
    nks = shape[:-2] # tuple defining number of k points in BZ

    orbs = model.get_orb() # ortbital vectors in reduced coordinates

    M = k_overlap_mat(u_wfs, orbs=orbs) # [kx, ky, b, m, n]

    ### minimizing Omega_tilde
    U, _ = find_min_unitary(model, M, iter_num=iter_num, eps=eps, print_=print_)
    u_max_loc = np.zeros(shape, dtype=complex)

    k_vals = [range(nk) for nk in nks] # list of dim(dim_k) with all k indices along each dir
    k_idx_arr = list(product(*k_vals)) # all pairwise combinations of k_indices
    for k in k_idx_arr:
        for i in range(u_wfs.shape[-2]):
            for j in range(u_wfs.shape[-2]):
                u_max_loc[k][i, :] += U[k][j, i] * u_wfs[k][j, :]

    return u_max_loc

def max_loc_Wan(
        model, u_wfs, tf_list, outer_states, 
        iter_num_omega_i = 3000, iter_num_omega_til=5000, eps=1e-3,
        state_idx=None, print_=False, return_uwfs=False
        ):
    """
    Find the maximally localized Wannier functions using the projection method

    Args:
        u_wfs: Bloch eigenstates defined over full k-mesh (excluding endpoint)
        tf_list: list of trial orbital sites and their associated weights (can be un-normalized)
        outer_states: manifold to 
    """
    if isinstance(u_wfs, wf_array):
        shape = u_wfs._wfs.shape # [*nks, idx, orb]
    else:
        shape = u_wfs.shape # [*nks, idx, orb]
    
    nks = shape[:-2] # tuple defining number of k points in BZ
    orbs = model.get_orb() # ortbital vectors in reduced coordinates
    # n_orb = model.get_num_orbitals() # number of orbitals
    # n_occ = int(n_orb/2) # number of occupied orbitals
    # lat_vecs = model.get_lat() # lattice vectors

    # get Bloch wfs by adding phase factors
    k_mesh = gen_k_mesh(*nks, flat=True, endpoint=False)
    psi_wfs = get_bloch_wfs(model, u_wfs, k_mesh)

    # Get initial tilde states from projection of trial wfs onto states spanned by the band indices specified
    psi_tilde = get_psi_tilde(psi_wfs, tf_list, state_idx=state_idx)
    u_tilde_wan = get_bloch_wfs(model, psi_tilde, k_mesh, inverse=True)

    M1 = k_overlap_mat(u_tilde_wan, orbs=orbs) # [kx, ky, b, m, n]
    spread, _, _ = spread_recip(model, M1, decomp=True)
    print(rf"Spread after initial projection = {spread[0]}")
    print(rf"Omega_I after initial projection = {spread[1]}")
    print(rf"Omega_til after initial projection = {spread[2]}")
    print()

    ### minimizing Omega_I
    util_min_Wan = find_optimal_subspace(
        model, outer_states, u_tilde_wan, iter_num=iter_num_omega_i, print_=print_)
    psi_til_min = get_bloch_wfs(model, util_min_Wan, k_mesh)
    # second projection of trial wfs onto full manifold spanned by psi_tilde
    psi_til_til_min = get_psi_tilde(psi_til_min, tf_list, state_idx=list(range(psi_til_min.shape[2])))
    u_til_til_min = get_bloch_wfs(model, psi_til_til_min, k_mesh, inverse=True)

    M2 = k_overlap_mat(u_til_til_min, orbs=orbs) # [kx, ky, b, m, n]
    spread, _, _ = spread_recip(model, M2, decomp=True)
    print(rf"Spread after minimizing Omega_I + 2nd proj = {spread[0]}")
    print(rf"Omega_I after minimizing Omega_I + 2nd proj = {spread[1]}")
    print(rf"Omega_til after minimizing Omega_I + 2nd proj = {spread[2]}")
    print()

    ### minimizing Omega_tilde
    # U, _ = find_min_unitary(model, M2, iter_num=iter_num_omega_til, eps=eps, print_=print_)
    # u_max_loc = np.zeros(u_til_til_min.shape, dtype=complex)
    # nkx, nky = nks[0], nks[1]
    # for kx in range(nkx):
    #     for ky in range(nky):
    #         for i in range(u_min.shape[2]):
    #             for j in range(u_min.shape[2]):
    #                 u_max_loc[kx, ky, i, :] += U[kx, ky, j, i] * u_til_til_min[kx, ky, j]

    u_max_loc = get_max_loc_uwfs(
        model, u_til_til_min, eps=eps, iter_num=iter_num_omega_til, print_=print_
        ) 
    psi_max_loc = get_bloch_wfs(model, u_max_loc, k_mesh, inverse=False)

    M3 = k_overlap_mat(u_max_loc, orbs=orbs) # [kx, ky, b, m, n]
    spread, _, _ = spread_recip(model, M3, decomp=True)
    print(rf"Spread after maximal localization = {spread[0]}")
    print(rf"Omega_I after maximal localization = {spread[1]}")
    print(rf"Omega_til after maximal localization = {spread[2]}")

    # Fourier transform Bloch-like states
    w0 = DFT(psi_max_loc)

    if return_uwfs:
        return w0, u_max_loc

    else:
        return w0