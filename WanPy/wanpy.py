import numpy as np
from itertools import product
from itertools import combinations_with_replacement as comb


def get_recip_lat_vecs(lat_vecs):
    b = 2 * np.pi * np.linalg.inv(lat_vecs).T
    return b


def get_k_shell(*nks, lat_vecs, N_sh, tol_dp=8, report=False):
    recip_vecs = get_recip_lat_vecs(lat_vecs)
    dk = np.array([recip_vecs[i] / nk for i, nk in enumerate(nks)])

    # vectors of integers multiplying dk
    nnbr_idx = list(product(list(range(-N_sh, N_sh + 1)), repeat=len(nks)))
    nnbr_idx.remove((0, 0))
    nnbr_idx = np.array(nnbr_idx)

    # vectors connecting k-points near Gamma point
    b_vecs = np.array([nnbr_idx[i] @ dk for i in range(nnbr_idx.shape[0])])
    dists = np.array([np.vdot(b_vecs[i], b_vecs[i]) for i in range(b_vecs.shape[0])])
    dists = dists.round(tol_dp)

    # sorting by distance
    sorted_idxs = np.argsort(dists)
    dists_sorted = dists[sorted_idxs]
    b_vecs_sorted = b_vecs[sorted_idxs]
    nnbr_idx_sorted = nnbr_idx[sorted_idxs]

    # keep only b_vecs in N_sh shells
    unique_dists = sorted(list(set(dists)))
    keep_dists = unique_dists[:N_sh]
    k_shell = [
        b_vecs_sorted[np.isin(dists_sorted, keep_dists[i])]
        for i in range(len(keep_dists))
    ]
    idx_shell = [
        nnbr_idx_sorted[np.isin(dists_sorted, keep_dists[i])]
        for i in range(len(keep_dists))
    ]

    if report:
        dist_degen = {ud: len(k_shell[i]) for i, ud in enumerate(keep_dists)}
        print("k-shell report:")
        print("--------------")
        print(f"Reciprocal lattice vectors: {recip_vecs}")
        print(f"Distances and degeneracies: {dist_degen}")
        print(f"k-shells: {k_shell}")
        print(f"idx-shells: {idx_shell}")

    return k_shell, idx_shell


def get_weights(*nks, lat_vecs, N_sh=1, report=False):
    k_shell, idx_shell = get_k_shell(*nks, lat_vecs=lat_vecs, N_sh=N_sh, report=report)
    dim_k = len(nks)
    Cart_idx = list(comb(range(dim_k), 2))
    n_comb = len(Cart_idx)

    A = np.zeros((n_comb, N_sh))
    q = np.zeros((n_comb))

    for j, (alpha, beta) in enumerate(Cart_idx):
        if alpha == beta:
            q[j] = 1
        for s in range(N_sh):
            b_star = k_shell[s]
            for i in range(b_star.shape[0]):
                b = b_star[i]
                A[j, s] += b[alpha] * b[beta]

    U, D, Vt = np.linalg.svd(A, full_matrices=False)
    w = (Vt.T @ np.linalg.inv(np.diag(D)) @ U.T) @ q
    if report:
        print(f"Finite difference weights: {w}")
    return w, k_shell, idx_shell


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
    end_pts = [-0.5, 0.5] if centered else [0, 1]

    k_vals = [np.linspace(end_pts[0], end_pts[1], nk, endpoint=endpoint) for nk in nks]
    mesh = np.array(list(product(*k_vals)))

    return mesh if flat else mesh.reshape(*[nk for nk in nks], len(nks))


def get_boundary_phase(*nks, orbs, idx_shell):
    k_idx_arr = list(
        product(*[range(nk) for nk in nks])
    )  # all pairwise combinations of k_indices

    bc_phase = np.ones((*nks, idx_shell[0].shape[0], orbs.shape[0]), dtype=complex)

    for k_idx_idx, k_idx in enumerate(k_idx_arr):
        for shell_idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
            k_nbr_idx = np.array(k_idx) + idx_vec
            # apply pbc to index
            mod_idx = np.mod(k_nbr_idx, nks)
            diff = k_nbr_idx - mod_idx
            G = np.divide(np.array(diff), np.array(nks))
            # if the translated k-index contains the -1st or last_idx+1 then we crossed the BZ boundary
            cross_bndry = np.any((k_nbr_idx == -1) | np.logical_or.reduce([k_nbr_idx == nk for nk in nks]))
            if cross_bndry:
                bc_phase[k_idx][shell_idx]= np.exp(-1j * 2 * np.pi * orbs @ G.T).T

    return bc_phase


def get_orb_phases(orbs, k_vec, inverse=False):
    """
    Introduces e^i{k.tau} factors

    Args:
        orbs (np.array): Orbital positions
        k_vec (np.array): k space grid (assumes flattened)
        inverse (boolean): whether to get cell periodic (True) or Bloch (False) wfs

    Returns:
      orb_phases (np.array): array of phases at each k value
    """
    lam = -1 if inverse else 1  # overall minus if getting cell periodic from Bloch
    per_dir = list(range(k_vec.shape[-1]))  # list of periodic dimensions
    # slice second dimension to only keep only periodic dimensions in orb
    per_orb = orbs[:, per_dir]

    # compute a list of phase factors [k_val, orbital]
    wf_phases = np.exp(lam * 1j * 2 * np.pi * per_orb @ k_vec.T, dtype=complex).T
    return wf_phases  # 1D numpy array of dimension norb


def get_bloch_wfs(orbs, u_wfs, k_mesh, inverse=False):
    """
    Change the cell periodic wfs to Bloch wfs

    Args:
    orbs (np.array): Orbital positions
    wfs (): cell periodic wfs [k, nband, norb]
    k_mesh (np.array): k-mesh on which u_wfs is defined

    Returns:
    wfs_psi: np.array
        wfs with orbitals multiplied by proper phase factor

    """
    shape = u_wfs.shape  # [*nks, idx, orb]
    nks = shape[:-2]
    norb = shape[-1]  # number of orbitals

    if len(k_mesh.shape) > 2:
        k_mesh = k_mesh.reshape(np.prod(nks), len(nks))  # flatten

    # Phases come in a list flattened over k space
    # Needs be reshaped to match k indexing of wfs
    phases = get_orb_phases(orbs, k_mesh, inverse=inverse).reshape(*nks, norb)
    # Broadcasting the phases to match dimensions
    psi_wfs = u_wfs * phases[..., np.newaxis, :] 

    return psi_wfs

def gen_rand_tf_list(n_tfs: int, n_orb: int):
    def gram_schmidt(vectors):
        orthogonal_vectors = []
        for v in vectors:
            for u in orthogonal_vectors:
                v -= np.dot(v, u) * u
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                orthogonal_vectors.append(v / norm)
        return np.array(orthogonal_vectors)

    # Generate three random 4-dimensional vectors
    vectors = abs(np.random.randn(n_tfs, n_orb))
    # Apply the Gram-Schmidt process to orthogonalize them
    orthonorm_vecs = gram_schmidt(vectors)

    tf_list = []

    for n in range(n_tfs):
        tf = []
        for orb in range(n_orb):
            tf.append((orb, orthonorm_vecs[n, orb]))

        tf_list.append(tf)

    return tf_list


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
            tfs[j, :] /= np.sqrt(sum(abs(tfs[j, :])))
        else:
            raise TypeError("tf_list is not of apporpriate type")

    # return numpy array containing trial functions
    return tfs  # tfs in order[trial funcs, orbitals]


def tf_overlap_mat(psi_wfs, tfs, state_idx):
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


def get_psi_tilde(psi_wf, tf_list, state_idx=None, compact_SVD=False):

    shape = psi_wf.shape
    n_orb = shape[-1]
    n_state = shape[-2]
    nks = shape[:-2]
    n_occ = int(n_state / 2)  # assuming half filled

    if state_idx is None:  # assume we are Wannierizing occupied bands at half-filling
        state_idx = list(range(0, n_occ))

    tfs = set_trial_function(tf_list, n_orb)
    A = tf_overlap_mat(psi_wf, tfs, state_idx)
    V, _, Wh = SVD(A, full_matrices=False, compact_SVD=compact_SVD)

    # swap only last two indices in transpose (ignore k indices)
    # slice psi_wf to keep only occupied bands
    psi_tilde = (V @ Wh).transpose(
        *([i for i in range(len(nks))] + [len(nks) + 1, len(nks)])
    ) @ psi_wf[..., state_idx, :]  # [*nk_i, nband, norb]

    return psi_tilde


def SVD(A, full_matrices=False, compact_SVD=False):
    # SVD on last 2 axes by default (preserving k indices)
    V, S, Wh = np.linalg.svd(A, full_matrices=full_matrices)

    # TODO: Test this method
    if compact_SVD: 
        V, S, Wh = np.linalg.svd(A, full_matrices=True)
        V = V[..., :, :-1]
        S = S[..., :-1]
        Wh = Wh[..., :-1, :]

    return V, S, Wh


def DFT(psi_tilde, norm=None):
    dim_k = len(psi_tilde.shape[:-2])
    Rn = np.fft.ifftn(psi_tilde, axes=[i for i in range(dim_k)], norm=norm)
    return Rn


def Wannierize(
    orbs,
    u_wfs,
    tf_list,
    state_idx=None,
    k_mesh=None,
    compact_SVD=False,
    ret_psi_til=False,
):
    """
    Obtains Wannier functions cenetered in home unit cell.

    Args:
        orbs (np.array): Orbital positions
        u_wfs (np.ndarray): wf array defined k-mesh excluding endpoints.
        tf_list (list): list of sites and amplitudes of trial wfs
        n_occ (int): number of occupied states to Wannierize from

        compact_SVD (bool, optional): For testing purposes. Defaults to False.
        switch_rep (bool, optional): For testing purposes. Defaults to False.
        tfs_swap (list, optional): For testing purposes. Defaults to None.

    Returns:
        w_0n (np.array): Wannier functions in home unit cell
    """
    
    shape = u_wfs.shape  # [*nks, idx, orb]
    nks = shape[:-2]

    # get Bloch wfs
    if k_mesh is None:  # assume u_wfs is defined over full BZ
        k_mesh = gen_k_mesh(*nks, flat=True, endpoint=False)
    psi_wfs = get_bloch_wfs(orbs, u_wfs, k_mesh)

    # get tilde states
    psi_tilde = get_psi_tilde(
        psi_wfs, tf_list, state_idx=state_idx, compact_SVD=compact_SVD
    )
    # u_tilde_wan = get_bloch_wfs(orbs, psi_tilde, k_mesh, inverse=True)

    # get Wannier functions
    w_0n = DFT(psi_tilde)

    if ret_psi_til:
        return w_0n, psi_tilde
    return w_0n


#### Computing Spread ######


# TODO: Allow for arbitrary dimensions and optimize
def spread_real(lat_vecs, orbs, w0, decomp=False):
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
    # shape = w0.shape # [*nks, idx, orb]
    # nxs = shape[:-2]
    # n_orb = shape[-1]
    # n_states = shape[-2]
    # assuming 2D for now
    nx, ny, n_wfs = w0.shape[0], w0.shape[1], w0.shape[2]
    # translation vectors in reduced units
    supercell = [
        (i, j) for i in range(-nx // 2, nx // 2) for j in range(-ny // 2, ny // 2)
    ]

    r_n = np.zeros((n_wfs, 2), dtype=complex)  # <\vec{r}>_n
    rsq_n = np.zeros(n_wfs, dtype=complex)  # <r^2>_n
    R_nm = np.zeros((2, n_wfs, n_wfs, nx * ny), dtype=complex)

    expc_rsq = 0  # <r^2>
    expc_r_sq = 0  # <\vec{r}>^2

    for n in range(n_wfs):  # "band" index
        for tx, ty in supercell:  # cells in supercell
            for i, orb in enumerate(orbs):  # values of Wannier function on lattice
                pos = (orb[0] + tx) * lat_vecs[0] + (orb[1] + ty) * lat_vecs[1]  # position
                r = np.sqrt(pos[0] ** 2 + pos[1] ** 2)

                w0n_r = w0[tx, ty, n, i]  # Wannier function

                # expectation value of position (vector)
                r_n[n, :] += abs(w0n_r) ** 2 * pos
                rsq_n[n] += r**2 * w0n_r * w0n_r.conj()

                if decomp:
                    for m in range(n_wfs):
                        for j, [dx, dy] in enumerate(supercell):
                            wRm_r = w0[
                                (tx + dx) % nx, (ty + dy) % ny, m, i
                            ]  # translated Wannier function
                            R_nm[:, n, m, j] += w0n_r * wRm_r.conj() * np.array(pos)

        expc_rsq += rsq_n[n]
        expc_r_sq += np.vdot(r_n[n, :], r_n[n, :])

    spread = expc_rsq - expc_r_sq

    if decomp:
        sigma_Rnm_sq = np.sum(np.abs(R_nm) ** 2)
        Omega_inv = expc_rsq - sigma_Rnm_sq
        Omega_tilde = sigma_Rnm_sq - np.sum(
            np.abs(
                np.diagonal(R_nm[:, :, :, supercell.index((0, 0))], axis1=1, axis2=2)
            )** 2
        )

        assert np.allclose(spread, Omega_inv + Omega_tilde)
        return [spread, Omega_inv, Omega_tilde], r_n, rsq_n

    else:
        return spread, r_n, rsq_n


def k_overlap_mat(lat_vecs, orbs, u_wfs):
    """
    Compute the overlap matrix of Bloch eigenstates. Assumes that the last u_wf
    along each periodic direction corresponds to the next to last k-point in the
    mesh (excludes endpoints). This way, the periodic boundary conditions are handled
    internally.

    Args:
        u_wfs (np.ndarray): The cell periodic Bloch wavefunctions
        orbs (np.ndarray): The orbitals positions
    Returns:
        M (np.array): overlap matrix
    """
   
    shape = u_wfs.shape  # [*nks, idx, orb]
    nks = shape[:-2]
    n_states = shape[-2]

    # Assumes only one shell for now
    _, idx_shell = get_k_shell(*nks, lat_vecs=lat_vecs, N_sh=1, tol_dp=8, report=False)
    bc_phase = get_boundary_phase(*nks, orbs=orbs, idx_shell=idx_shell)

    # assumes that there is no last element in the k mesh, so we need to introduce phases
    M = np.zeros(
        (*nks, len(idx_shell[0]), n_states, n_states), dtype=complex
    )  # overlap matrix
    for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
        states_pbc = np.roll(u_wfs, shift=tuple(-idx_vec), axis=(0,1)) * bc_phase[..., idx, np.newaxis,  :]
        M[..., idx, :, :] = np.einsum("...mj, ...nj -> ...mn", u_wfs.conj(), states_pbc)
    return M


def get_Omega_til(M, w_b, k_shell):
    nks = M.shape[:-3]
    Nk = np.prod(nks)
    k_axes = tuple([i for i in range(len(nks))])

    diag_M = np.diagonal(M, axis1=-1, axis2=-2)
    log_diag_M_imag = np.log(diag_M).imag
    abs_diag_M_sq = abs(diag_M) ** 2

    r_n = -(1 / Nk) * w_b * np.sum(log_diag_M_imag, axis=k_axes).T @ k_shell

    Omega_tilde = (1 / Nk) * w_b * ( 
            np.sum((-log_diag_M_imag - k_shell @ r_n.T)**2) + 
            np.sum(abs(M)**2) - np.sum(abs_diag_M_sq)
        )
    return Omega_tilde


def get_Omega_I(M, w_b, k_shell):
    
    Nk = np.prod(M.shape[:-3])
    n_states = M.shape[3]
    Omega_i = w_b * n_states * k_shell.shape[0] - (1 / Nk) * w_b * np.sum(abs(M) **2)

    return Omega_i


def spread_recip(lat_vecs, M, decomp=False):
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
    n_states = shape[3]
    nks = shape[:-3]
    k_axes = tuple([i for i in range(len(nks))])
    Nk = np.prod(nks)

    w_b, k_shell, _ = get_weights(*nks, lat_vecs=lat_vecs, N_sh=1)
    w_b, k_shell = w_b[0], k_shell[0] # Assume only one shell for now

    diag_M = np.diagonal(M, axis1=-1, axis2=-2)
    log_diag_M_imag = np.log(diag_M).imag
    abs_diag_M_sq = abs(diag_M) ** 2

    r_n = -(1 / Nk) * w_b * np.sum(log_diag_M_imag, axis=k_axes).T @ k_shell
    rsq_n = (1 / Nk) * w_b * np.sum(
        (1 - abs_diag_M_sq + log_diag_M_imag ** 2), axis=k_axes+tuple([-2]))
    
    spread_n = rsq_n - np.array([np.vdot(r_n[n, :], r_n[n, :]) for n in range(r_n.shape[0])])
    # expc_rsq = np.sum(rsq_n)  # <r^2>
    # expc_r_sq = np.sum([np.vdot(r_n[n, :], r_n[n, :]) for n in range(r_n.shape[0])])  # <\vec{r}>^2
    # spread = expc_rsq - expc_r_sq
    if decomp:
        Omega_i = w_b * n_states * k_shell.shape[0] - (1 / Nk) * w_b * np.sum(abs(M) **2)
        Omega_tilde = (1 / Nk) * w_b * ( 
            np.sum((-log_diag_M_imag - k_shell @ r_n.T)**2) + 
            np.sum(abs(M)**2) - np.sum(abs_diag_M_sq)
        )
        return [spread_n, Omega_i, Omega_tilde], r_n, rsq_n

    else:
        return spread_n, r_n, rsq_n
    

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
    phase = np.array(np.exp(-1j * 2 * np.pi * orbs @ np.array(G).T), dtype=complex).T
    return phase


def swap_reps(eigvecs, k_points, swap_pts, swap_scheme):
    swap_eigvecs = eigvecs.copy()
    nks = eigvecs.shape[:-2]

    diff = np.linalg.norm(k_points - np.array(swap_pts), axis=len(nks))
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
        eigvecs (np.ndarray):
            Eigenvectors spanning the target subspace
        k_path (np.array):
            1D path on which we want to diagonalize the Hamiltonian

    Returns:
        eigvals (np.array):
            eigenvalues in subspace
    """
   
    shape = eigvecs.shape  # [*nks, idx, orb]
    nks = shape[:-2]
    n_orb = shape[-1]
    n_states = shape[-2]

    H_k_proj = np.zeros([*nks, n_states, n_states], dtype=complex)

    for k_idx, k in enumerate(k_path):
        H_k = model._gen_ham(k)
        V = np.transpose(eigvecs[k_idx], axes=(1, 0))  # [orb, num_evecs]
        H_k_proj[k_idx, :, :] = V.conj().T @ H_k @ V  # projected Hamiltonian

    eigvals = np.zeros((*nks, n_states), dtype=complex)
    evecs = np.zeros((*nks, n_states, n_orb), dtype=complex)

    for idx, k in enumerate(k_path):
        eigvals[idx, :], evec = np.linalg.eigh(H_k_proj[idx])  # [k, n], [evec wt, n]
        for i in range(evec.shape[1]):
            # Returns in given eigvec basis
            evecs[idx, i, :] = sum(
                [evec[j, i] * eigvecs[idx, j, :] for j in range(evec.shape[0])]
            )

    if ret_evecs:
        return eigvals.real, evecs
    else:
        return eigvals.real


####### Maximally Localized WF ############

def find_optimal_subspace(
    lat_vecs, orbs, outer_states, inner_states, iter_num=100, verbose=False, tol=1e-17, alpha=1
):
    shape = inner_states.shape  # [*nks, idx, orb]
    nks = shape[:-2]
    Nk = np.prod(nks)
    n_orb = shape[-1]
    n_states = shape[-2]
    dim_subspace = n_states
   
    # Assumes only one shell for now
    w_b, _, idx_shell = get_weights(*nks, lat_vecs=lat_vecs, N_sh=1)
    num_nnbrs = len(idx_shell[0])
    bc_phase = get_boundary_phase(*nks, orbs=orbs, idx_shell=idx_shell)

    P = np.einsum("...ni, ...nj->...ij", inner_states, inner_states.conj())

    # Projector on initial subspace at each k (for pbc of neighboring spaces)
    P_nbr = np.zeros((*nks, num_nnbrs, n_orb, n_orb), dtype=complex)
    Q_nbr = np.zeros((*nks, num_nnbrs, n_orb, n_orb), dtype=complex)
    T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)

    for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
        states_pbc = np.roll(inner_states, shift=tuple(-idx_vec), axis=(0,1)) * bc_phase[..., idx, np.newaxis,  :]
        P_nbr[..., idx, :, :] = np.einsum(
                "...ni, ...nj->...ij", states_pbc, states_pbc.conj()
                )
        Q_nbr[..., idx, :, :] = np.eye(n_orb) - P_nbr[..., idx, :, :]
        T_kb[..., idx] = np.trace(P[..., :, :] @ Q_nbr[..., idx, :, :], axis1=-1, axis2=-2)

    P_min = np.copy(P)  # start of iteration
    P_nbr_min = np.copy(P_nbr)  # start of iteration
    Q_nbr_min = np.copy(Q_nbr)  # start of iteration

    # states spanning optimal subspace minimizing gauge invariant spread
    states_min = np.zeros((*nks, dim_subspace, n_orb), dtype=complex)
    omega_I_prev = (1 / Nk) * w_b[0] * np.sum(T_kb)

    for i in range(iter_num):
        P_avg = np.sum(w_b[0] * P_nbr_min, axis=-3)
        Z = outer_states[..., :, :].conj() @ P_avg @ np.transpose(outer_states[..., : ,:], axes=(0,1,3,2))

        _, eigvecs = np.linalg.eigh(Z)  # [val, idx]
        states_min = np.einsum('...ij, ...ik->...jk', eigvecs[..., -dim_subspace:], outer_states)

        P_new = np.einsum("...ni,...nj->...ij", states_min, states_min.conj())
        P_min = alpha * P_new + (1 - alpha) * P_min # for next iteration
        
        for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
            states_pbc = np.roll(states_min, shift=tuple(-idx_vec), axis=(0,1)) * bc_phase[..., idx, np.newaxis,  :]
            P_nbr_min[..., idx, :, :] = np.einsum(
                    "...ni, ...nj->...ij", states_pbc, states_pbc.conj()
                    )
            Q_nbr_min[..., idx, :, :] = np.eye(n_orb) - P_nbr_min[..., idx, :, :]
            T_kb[..., idx] = np.trace(P_min[..., :, :] @ Q_nbr_min[..., idx, :, :], axis1=-1, axis2=-2)
        
        omega_I_new = (1 / Nk) * w_b[0] * np.sum(T_kb)

        if omega_I_new > omega_I_prev:
            print("Warning: Omega_I is increasing.")

        if abs(omega_I_prev - omega_I_new) <= tol:
            print("Omega_I has converged within tolerance. Breaking loop")
            return states_min

        if verbose:
            print(f"{i} Omega_I: {omega_I_new.real}")

        omega_I_prev = omega_I_new

    return states_min


def mat_exp(M):
    eigvals, eigvecs = np.linalg.eig(M)
    U = eigvecs
    U_inv = np.linalg.inv(U)

    # Diagonal matrix of the exponentials of the eigenvalues
    exp_diagM = np.exp(eigvals)

    # Construct the matrix exponential
    expM = np.einsum('...ij,...jk->...ik', U, np.multiply(U_inv, exp_diagM[..., :, np.newaxis]))
    return expM

def find_min_unitary(u_wfs, lat_vecs, orbs, eps=1 / 160, iter_num=10, verbose=False, tol=1e-12):
    """
    Finds the unitary that minimizing the gauge dependent part of the spread. 

    Args:
        lat_vecs: Lattice vectors
        M: Overlap matrix
        eps: Step size for gradient descent
        iter_num: Number of iterations
        verbose: Whether to print the spread at each iteration
        tol: If difference of spread is lower that tol for consecutive iterations,
            the loop breaks

    Returns:
        U: The unitary matrix
        M: The rotated overlap matrix
    
    """
    M = k_overlap_mat(lat_vecs, orbs, u_wfs)  # [kx, ky, b, m, n]
    shape = M.shape
    nks = shape[:-3]
    # dim_k = len(nks)
    Nk = np.prod(nks)
    num_state = shape[-1]

    # Assumes only one shell for now
    w_b, k_shell, idx_shell = get_weights(*nks, lat_vecs=lat_vecs, N_sh=1)
    w_b, k_shell = w_b[0], k_shell[0]

    U = np.zeros((*nks, num_state, num_state), dtype=complex)  # unitary transformation
    U[...] = np.eye(num_state, dtype=complex)  # initialize as identity
    M0 = np.copy(M)  # initial overlap matrix
    M = np.copy(M)  # new overlap matrix

    # initializing
    grad_mag_prev = 0
    eta = 1
    for i in range(iter_num):
        log_diag_M_imag = np.log(np.diagonal(M, axis1=-1, axis2=-2)).imag
        r_n = -(1 / Nk) * w_b * np.sum(
            log_diag_M_imag, axis=(0,1)).T @ k_shell
        q = log_diag_M_imag + (k_shell @ r_n.T)
        R = np.multiply(M, np.diagonal(M, axis1=-1, axis2=-2)[..., np.newaxis, :].conj())
        T = np.multiply(np.divide(M, np.diagonal(M, axis1=-1, axis2=-2)[..., np.newaxis, :]), q[..., np.newaxis, :])
        A_R = (R - np.transpose(R, axes=(0,1,2,4,3)).conj()) / 2
        S_T = (T + np.transpose(T, axes=(0,1,2,4,3)).conj()) / (2j)
        G = 4 * w_b * np.sum(A_R - S_T, axis=-3)
        # G = optimizer.update(G)
        U = np.einsum("...ij, ...jk -> ...ik", U, mat_exp(eta * eps * G))

        U_conj_trans = np.transpose(U, axes=(0,1,3,2)).conj()
        for idx, idx_vec in enumerate(idx_shell[0]):
            U_shifted = np.roll(U, shift=tuple(-idx_vec), axis=(0,1))
            M[..., idx, :, :] = (
                U_conj_trans @  M0[..., idx, :, :]  @ U_shifted
                )

        grad_mag = np.linalg.norm(np.sum(G, axis=(0,1)))

        if abs(grad_mag) <= tol:
            print("Omega_tilde minimization has converged within tolerance. Breaking the loop")
            u_max_loc = np.einsum('...ji, ...jm -> ...im', U, u_wfs)
            return u_max_loc, U
        if grad_mag_prev < grad_mag and i!=0:
            print("Warning: Gradient increasing.")
            # # eta *= 0.9
            # scale = np.amax(U)
            # pert = np.random.normal(scale=scale*1e-3, size=U.shape) + 1j * np.random.normal(scale=scale*1e-3, size=U.shape)
            # pert = (pert + pert.swapaxes(-1, -2).conj())/2j
            # U = np.einsum("...ij, ...jk -> ...ik", U, mat_exp(pert))  # Perturb U to escape local minima
        if abs(grad_mag_prev - grad_mag) <= tol:
            print("Warning: Found local minima.")
        #     scale = np.amax(U)
        #     pert = np.random.normal(scale=scale*1e-3, size=U.shape)
        #     pert = (pert + pert.swapaxes(-1, -2).conj())/2j
        #     U = np.einsum("...ij, ...jk -> ...ik", U, mat_exp(pert))  # Perturb U to escape local minima
        
        # eta = max(eta * 0.99, 0.1)  # Decay eta but keep it above a threshold
        if verbose:
            omega_tilde = get_Omega_til(M, w_b, k_shell)
            print(
                f"{i} Omega_til = {omega_tilde.real}, Grad mag: {grad_mag}"
            )
       
        grad_mag_prev = grad_mag

    u_max_loc = np.einsum('...ji, ...jm -> ...im', U, u_wfs)
    return u_max_loc, U


def max_loc_Wan(
    lat_vecs,
    orbs,
    u_wfs,
    tf_list,
    outer_states,
    iter_num_omega_i=1000,
    iter_num_omega_til=1000,
    alpha=1,
    eps=1e-3,
    tol=1e-10,
    Wan_idxs=None,
    return_uwfs=False,
    return_wf_centers=False,
    return_spread=False,
    verbose=False,
    report=True,
):
    """
    Find the maximally localized Wannier functions using the projection method.

    Args:
        lat_vecs(np.ndarray): Lattice vectors
        orbs(np.ndarray): Orbital vectors
        u_wfs(np.ndarray): Bloch eigenstates defined over full k-mesh (excluding endpoint)
        tf_list(list): list of trial orbital sites and their associated weights (can be non-normalized)
        outer_states(np.ndarray): Disentanglement manifold 
        state_idx (list | None): Specifying the band indices of u_wfs to Wannierize via projection.
            By default, will assume half filled insulator and Wannierize the lower
            half of the bands.
        return_uwfs(bool): Whether to return the Bloch states corresponding to maximally localized 
            Wannier functions
        return_wf_centers(bool): Whether to return the positions of the Wannier function centers
        verbose(bool): Whether to print spread during minimization
        report(bool): Whether to print the final spread and Wannier centers

    """
    
    shape = u_wfs.shape  # [*nks, idx, orb]
    nks = shape[:-2]  # tuple defining number of k points in BZ

    # get Bloch wfs by adding phase factors
    k_mesh = gen_k_mesh(*nks, flat=True, endpoint=False)
    psi_wfs = get_bloch_wfs(orbs, u_wfs, k_mesh)

    # Get tilde states from initial projection of trial wfs onto states spanned by the band indices specified
    psi_tilde = get_psi_tilde(psi_wfs, tf_list, state_idx=Wan_idxs)
    u_tilde_wan = get_bloch_wfs(orbs, psi_tilde, k_mesh, inverse=True)

    # Minimizing Omega_I via disentanglement
    util_min_Wan = find_optimal_subspace(
        lat_vecs,
        orbs,
        outer_states,
        u_tilde_wan,
        iter_num=iter_num_omega_i,
        verbose=verbose, alpha=alpha, tol=tol
    )
    psi_til_min = get_bloch_wfs(orbs, util_min_Wan, k_mesh)

    # Second projection of trial wfs onto full manifold spanned by psi_tilde
    psi_til_til_min = get_psi_tilde(
        psi_til_min, tf_list, state_idx=list(range(psi_til_min.shape[2]))
    )
    u_til_til_min = get_bloch_wfs(orbs, psi_til_til_min, k_mesh, inverse=True)

    # Optimal gauge selection
    u_max_loc, _ = find_min_unitary(u_til_til_min, lat_vecs, orbs, eps=eps, iter_num=iter_num_omega_til, verbose=verbose, tol=tol)
    psi_max_loc = get_bloch_wfs(orbs, u_max_loc, k_mesh, inverse=False)

    # Fourier transform Bloch-like states
    w0 = DFT(psi_max_loc)

    M = k_overlap_mat(lat_vecs, orbs, u_max_loc)  # [kx, ky, b, m, n]
    spread, expc_r, expc_rsq = spread_recip(lat_vecs, M, decomp=True)

    if report:
        print("Post processing report:")
        print(" --------------- ")
        print(rf"Quadratic spread = {spread[0]}")
        print(rf"Omega_i = {spread[1]}")
        print(rf"Omega_tilde = {spread[2]}")
        print(f"<\\vec{{r}}>_n = {expc_r}")
        print(f"<r^2>_n = {expc_rsq}")

    ret_pckg = [w0]
    if return_uwfs:
        ret_pckg.append(u_max_loc)
    if return_wf_centers:
        ret_pckg.append(expc_r)
    if return_spread:
        ret_pckg.append(spread)
    return ret_pckg
