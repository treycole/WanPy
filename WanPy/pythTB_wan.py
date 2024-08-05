from pythtb import *
from typing import TYPE_CHECKING
import numpy as np
from itertools import product
from itertools import combinations_with_replacement as comb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


if TYPE_CHECKING:
    from pythtb import tb_model, wf_array


class Lattice():
    def __init__(self, model: tb_model):
        self._orbs = model.get_orb()
        self._n_orb = model.get_num_orbitals()
        self._lat_vecs = model.get_lat() # lattice vectors
        self._recip_lat_vecs = self.get_recip_lat_vecs()

    def report(self):
        """Prints information about the lattice attributes."""
        print("Lattice vectors (Cartesian coordinates)")
        for idx, lat_vec in enumerate(self._lat_vecs):
            print(f"a_{idx} ===> {lat_vec}")
        print("Reciprocal lattice vectors (1/Cartesian coordinates)")
        for idx, recip_lat_vec in enumerate(self._recip_lat_vecs):
            print(f"b_{idx} ===> {recip_lat_vec}")
        orb_pos = self._orbs @ self._lat_vecs
        print("Position of orbitals (Cartesian)")
        for idx, pos in enumerate(orb_pos):
            print(f"{idx} ===> {pos}")
        print("Position of orbitals (reduced coordinates)")
        for idx, pos in enumerate(self._orbs):
            print(f"{idx} ===> {pos}")

    def get_recip_lat_vecs(self):
        """Returns reciprocal lattice vectors."""
        b = 2 * np.pi * np.linalg.inv(self._lat_vecs).T
        return b
    
    def get_orb(self, Cartesian: bool = False):
        """Returns orbtial vectors."""
        if Cartesian:
            return self._orbs @ self._lat_vecs
        else:
            return self._orbs


class K_mesh():
    def __init__(self, model: tb_model, *nks):
        self.Lattice = Lattice(model)
        self.nks = nks
        self.idx_arr = list(product(*[range(nk) for nk in nks]))  # list of all k_indices
        self.full_mesh = self.gen_k_mesh(flat=False, endpoint=False)
        self.flat_mesh = self.gen_k_mesh(flat=True, endpoint=False)

    def gen_k_mesh(
            self, 
            centered: bool = False, 
            flat: bool = True, 
            endpoint: bool = False
            ) -> np.ndarray:
        """Generate k-mesh in reduced coordinates

        Args:
            centered: 
                If True, mesh is defined from [-0.5, 0.5] along each direction. 
                Defaults to False.
            flat:
                If True, returns flattened array of k-points. If False, returns
                reshaped array with axes along each k-space dimension. Defaults to True.
            endpoint: 
                If True, includes 1 in the mesh. Defaults to False.

        Returns:
            k-mesh: 
                Array of k-mesh coordinates.
        """
        end_pts = [-0.5, 0.5] if centered else [0, 1]
        # end_pts = [0, 1]

        k_vals = [np.linspace(end_pts[0], end_pts[1], nk, endpoint=endpoint) for nk in self.nks]
        mesh = np.array(list(product(*k_vals)))

        return mesh if flat else mesh.reshape(*[nk for nk in self.nks], len(self.nks))
    
    def get_k_shell(
            self, 
            N_sh: int, 
            report: bool = False
            ):
        """Generates shells of k-points around Gamma.

        Returns array of vectors connecting the origin to nearest neighboring k-points 
        in a given mesh, along with vectors of reduced coordinates. 

        Args:
            N_sh: 
                Number of nearest neighbor shells.
            report:
                If True, prints 
        
        """
        dk = np.array([self.Lattice._recip_lat_vecs[i] / nk for i, nk in enumerate(self.nks)])

        # vectors of integers multiplying dk
        nnbr_idx = list(product(list(range(-N_sh, N_sh + 1)), repeat=len(self.nks)))
        nnbr_idx.remove((0, 0))
        nnbr_idx = np.array(nnbr_idx)

        # vectors connecting k-points near Gamma point
        b_vecs = np.array([nnbr_idx[i] @ dk for i in range(nnbr_idx.shape[0])])
        dists = np.array([np.vdot(b_vecs[i], b_vecs[i]) for i in range(b_vecs.shape[0])])
        dists = dists.round(10)

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
            print(f"Reciprocal lattice vectors: {self._recip_vecs}")
            print(f"Distances and degeneracies: {dist_degen}")
            print(f"k-shells: {k_shell}")
            print(f"idx-shells: {idx_shell}")

        return k_shell, idx_shell
    
    
    def get_weights(self, N_sh=1, report=False):
        k_shell, idx_shell = self.get_k_shell(N_sh=N_sh, report=report)
        dim_k = len(self.nks)
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


class Bloch():
    def __init__(self, model: tb_model, *nks):
        self.model = model
        self.Lattice = Lattice(model)
        self.K_mesh = K_mesh(model, *nks)
        self.set_Bloch_ham()


    def solve_model(self):
        """
        Solves for the Bloch eigenstates on the system defined by the model over a semi-full 
        k-mesh, e.g. in 3D reduced coordinates {k = [kx, ky, kz] | k_i in [0, 1)}.
        """
        u_wfs = wf_array(self.model, [*self.K_mesh.nks])
        for k_idx in self.K_mesh.idx_arr:
            u_wfs.solve_on_one_point(self.K_mesh.full_mesh[k_idx], [*k_idx])
        u_wfs = np.array(u_wfs._wfs, dtype=complex)
        self.set_wfs(u_wfs)

    def get_states(self):
        """Returns dictionary containing Bloch and cell-periodic eigenstates."""
        assert hasattr(self, "_psi_wfs"), "Need to call `solve_model` or `set_wfs` to initialize Bloch states"
        return {"Bloch": self._psi_wfs, "Cell periodic": self._u_wfs}
    
    def get_Bloch_Ham(self):
        return self.H_k
    
    def get_overlap_mat(self):
        """Returns overlap matrix."""
        assert hasattr(self, "_M"), "Need to call `solve_model` or `set_wfs` to initialize overlap matrix"
        return self._M
    
    def get_orb_phases(self, inverse=False):
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
        per_dir = list(range(self.K_mesh.flat_mesh.shape[-1]))  # list of periodic dimensions
        # slice second dimension to only keep only periodic dimensions in orb
        per_orb = self.Lattice._orbs[:, per_dir]

        # compute a list of phase factors [k_val, orbital]
        wf_phases = np.exp(lam * 1j * 2 * np.pi * per_orb @ self.K_mesh.flat_mesh.T, dtype=complex).T
        return wf_phases  # 1D numpy array of dimension norb
    
    
    def set_Bloch_ham(self):
        H_k = np.zeros((*self.K_mesh.nks, self.Lattice._n_orb, self.Lattice._n_orb), dtype=complex)

        for k_idx in self.K_mesh.idx_arr:
            k_pt = self.K_mesh.full_mesh[k_idx]
            H_k[k_idx] = self.model._gen_ham(k_pt)

        self.H_k = H_k


    def set_wfs(self, wfs, cell_periodic=True):
        """
        Sets the Bloch and cell-periodic eigenstates as class attributes.

        Args:
            wfs (np.ndarray): 
                Bloch (or cell-periodic) eigenstates defined on a k-mesh corresponding
                to nks passed during class instantiation. The mesh is assumed to exlude the
                endpoints, e.g. in reduced coordinates {k = [kx, ky, kz] | k_i in [0, 1)}. 

        """
        if cell_periodic:
            self._u_wfs = wfs
            self._psi_wfs = self.apply_phase(wfs)
        else:
            self._psi_wfs = wfs
            self._u_wfs = self.apply_phase(wfs, inverse=True)

        self._n_states = self._u_wfs.shape[-2]
        self._M = self.k_overlap_mat()
    

    def apply_phase(self, wfs, inverse=False):
        """
        Change between cell periodic and Bloch wfs by multiplying exp(\pm i k . r)

        Args:
        wfs (pythtb.wf_array): Bloch or cell periodic wfs [k, nband, norb]

        Returns:
        wfsxphase (np.ndarray): 
            wfs with orbitals multiplied by phase factor

        """
        phases = self.get_orb_phases(inverse=inverse).reshape(*self.K_mesh.nks, self.Lattice._n_orb)
        # Broadcasting the phases to match dimensions
        wfsxphase = wfs * phases[..., np.newaxis, :] 
        return wfsxphase
    
    def get_pbc_phase(orbs, G):
        """
        Get phase factors to multiply the cell periodic states in the first BZ
        related by the pbc u_{n, k+G} = u_{n, k} exp(-i G . r)

        Args:
            orbs (np.ndarray): 
                reduced coordinates of orbital positions
            G (list): 
                reciprocal lattice vector in reduced coordinates connecting the 
                cell periodic states in different BZs

        Returns:
            phase (np.ndarray): 
                phase factor to be multiplied to the cell periodic eigenstates
                in first BZ
        """
        phase = np.array(np.exp(-1j * 2 * np.pi * orbs @ np.array(G).T), dtype=complex).T
        return phase

    def get_boundary_phase(self, idx_shell):
        """
        Get phase factors to multiply the cell periodic states in the first BZ
        related by the pbc u_{n, k+G} = u_{n, k} exp(-i G . r)

        Args:
            idx_shell (np.ndarray): 
                array of index vectors that connect nearrest neighbor k-points in
                the k-mesh array. The vector is to be added to the k_mesh index such 
                that k_mesh[i + idx_shell[j]] will take you to the nearest neighbor 
                in the k_mesh.

        Returns:
            bc_phase (np.ndarray): 
                The shape is [...k(s), shell_idx] where shell_idx is an integer
                corresponding to a particular idx_vec where the convention is to go  
                counter-clockwise (e.g. square lattice 0 --> [1, 0], 1 --> [0, 1] etc.)

        """
        k_idx_arr = list(
            product(*[range(nk) for nk in self.K_mesh.nks])
        )  # all pairwise combinations of k_indices

        bc_phase = np.ones((*self.K_mesh.nks, idx_shell[0].shape[0], self.Lattice._orbs.shape[0]), dtype=complex)

        for k_idx in k_idx_arr:
            for shell_idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
                k_nbr_idx = np.array(k_idx) + idx_vec
                # apply pbc to index
                mod_idx = np.mod(k_nbr_idx, self.K_mesh.nks)
                diff = k_nbr_idx - mod_idx
                G = np.divide(np.array(diff), np.array(self.K_mesh.nks))
                # if the translated k-index contains the -1st or last_idx+1 then we crossed the BZ boundary
                cross_bndry = np.any((k_nbr_idx == -1) | np.logical_or.reduce([k_nbr_idx == nk for nk in self.K_mesh.nks]))
                if cross_bndry:
                    bc_phase[k_idx][shell_idx]= np.exp(-1j * 2 * np.pi * self.Lattice._orbs @ G.T).T

        return bc_phase
    
    def k_overlap_mat(self):
        """
        Compute the overlap matrix of the cell periodic eigenstates. Assumes that the last u_wf
        along each periodic direction corresponds to the next to last k-point in the
        mesh (excludes endpoints). This way, the periodic boundary conditions are handled
        internally.

        Returns:
            M (np.array): overlap matrix
        """

        # Assumes only one shell for now
        _, idx_shell = self.K_mesh.get_k_shell(N_sh=1, report=False)
        bc_phase = self.get_boundary_phase(idx_shell)

        M = np.zeros(
            (*self.K_mesh.nks, len(idx_shell[0]), self._n_states, self._n_states), dtype=complex
        )  # overlap matrix
        
        for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
            # introduce phases to states when k+b is across the BZ boundary
            states_pbc = np.roll(self._u_wfs, shift=tuple(-idx_vec), axis=(0,1)) * bc_phase[..., idx, np.newaxis,  :]
            M[..., idx, :, :] = np.einsum("...mj, ...nj -> ...mn", self._u_wfs.conj(), states_pbc)
        return M
    

class Wannier():
    def __init__(
            self, model: tb_model, nks: list  
            ):
        self._model = model
        self._nks = nks

        self.Lattice = Lattice(model)
        self.K_mesh = K_mesh(model, *nks)

        self.energy_eigstates = Bloch(model, *nks)
        self.energy_eigstates.solve_model()
        self.tilde_states = Bloch(model, *nks)

    def get_tilde_states(self):
        return self.tilde_states.get_states()
    
    def get_Bloch_Ham(self):
        return self.tilde_states.get_Bloch_Ham()

    def gen_rand_tf_list(self, n_tfs: int):
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
        vectors = abs(np.random.randn(n_tfs, self.Lattice._n_orb))
        # Apply Gram-Schmidt to orthogonalize them
        orthonorm_vecs = gram_schmidt(vectors)

        tf_list = []
        for n in range(n_tfs):
            tf = []
            for orb in range(self.Lattice._n_orb):
                tf.append((orb, orthonorm_vecs[n, orb]))
            tf_list.append(tf)
        return tf_list
    

    def get_trial_wfs(self, tf_list):
        """
        Args:
            tf_list: list[int | list[tuple]]
                list of numbers or tuples defining either the integer site
                of the trial function (delta) or the tuples (site, amplitude)
    
        Returns:
            tfs (num_tf x norb np.array): 2 dimensional array of trial functions
        """
        # number of trial functions to define
        num_tf = len(tf_list)

        # initialize array containing tfs = "trial functions"
        tfs = np.zeros([num_tf, self.Lattice._n_orb], dtype=complex)

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
    
    
    def tf_overlap_mat(self, psi_wfs, tfs, state_idx):
        """

        Args:
            psi_wfs (np.array): Bloch eigenstates
            tfs (np.array): trial wfs
            state_idx (list): band indices to form overlap matrix with

        Returns:
            A (np.array): overlap matrix
        """

        ntfs = tfs.shape[0]
        A = np.zeros((*self.K_mesh.nks, len(state_idx), ntfs), dtype=complex)
        for n in state_idx:
            for j in range(ntfs):
                A[..., n, j] = psi_wfs.conj()[..., n, :] @ tfs[j, :]

        return A
    
    
    def get_psi_tilde(self, psi_wfs, tfs, state_idx=None, compact_SVD=False):
        A = self.tf_overlap_mat(psi_wfs, tfs, state_idx=state_idx)
        # TODO: Test this method
        if compact_SVD: 
            V, S, Wh = np.linalg.svd(A, full_matrices=True)
            V = V[..., :, :-1]
            S = S[..., :-1]
            Wh = Wh[..., :-1, :]

        V, S, Wh = np.linalg.svd(A, full_matrices=False)

        # swap only last two indices in transpose (ignore k indices)
        # slice psi_wf to keep only occupied bands
        psi_tilde = (V @ Wh).transpose(
            *([i for i in range(len(self.K_mesh.nks))] + [len(self.K_mesh.nks) + 1, len(self.K_mesh.nks)])
        ) @ psi_wfs[..., state_idx, :]  # [*nk_i, nband, norb]

        return psi_tilde
    
    
    def Wannierize(self, tf_list: list, band_idxs: list | None =None):
        """
        Obtains Wannier functions cenetered in home unit cell.

        Args:
            tf_list (list): List of tuples with sites and weights. Can be un-normalized. 
            band_idxs (list | None): Band indices to Wannierize. Defaults to occupied bands (lower half).

        Returns:
            w_0n (np.array): Wannier functions in home unit cell
        """

        self.tf_list = tf_list
        if tf_list[0] == "random":
            n_tfs = tf_list[1]
            tf_list = self.gen_rand_tf_list(n_tfs)
        self.trial_wfs = self.get_trial_wfs(tf_list)

        if band_idxs is None:  # assume we are Wannierizing occupied bands at half-filling
            n_occ = int(self.energy_eigstates._n_states / 2)  # assuming half filled
            band_idxs = list(range(0, n_occ))

        psi_tilde = self.get_psi_tilde(self.energy_eigstates._psi_wfs, self.trial_wfs, state_idx=band_idxs)
        self.tilde_states.set_wfs(psi_tilde, cell_periodic=False)

        psi_wfs = self.tilde_states._psi_wfs
        dim_k = len(psi_wfs.shape[:-2])
        # DFT
        self.WFs = np.fft.ifftn(psi_wfs, axes=[i for i in range(dim_k)], norm=None)

        spread = self.spread_recip(decomp=True)
        self.spread = spread[0][0]
        self.omega_i = spread[0][1]
        self.omega_til = spread[0][2]
        self.centers = spread[1]

    
    # TODO: Allow for arbitrary dimensions and optimize
    def spread_real(self, decomp=False):
        """
        Spread functional computed in real space with Wannier functions

        Args:
            decomp (boolean): whether to separate gauge (in)variant parts of spread

        Returns:
            Omega: the spread functional
            Omega_inv: (optional) the gauge invariant part of the spread
            Omega_tilde: (optional) the gauge dependent part of the spread
            expc_rsq: <r^2>_{n}
            expc_r_sq: <\vec{r}>_{n}^2
        """
        w0 = self.WFs
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
                for i, orb in enumerate(self.Lattice._orbs):  # values of Wannier function on lattice
                    # ( orb + vec(t) ) @ lat_vecs
                    pos = (orb[0] + tx) * self.Lattice._lat_vecs[0] + (orb[1] + ty) * self.Lattice._lat_vecs[1]  # position
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


    def spread_recip(self, decomp=False):
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
        M = self.tilde_states._M
        w_b, k_shell, _ = self.K_mesh.get_weights()
        w_b, k_shell = w_b[0], k_shell[0] # Assume only one shell for now

        shape = M.shape
        n_states = shape[3]
        nks = shape[:-3]
        k_axes = tuple([i for i in range(len(nks))])
        Nk = np.prod(nks)

        diag_M = np.diagonal(M, axis1=-1, axis2=-2)
        log_diag_M_imag = np.log(diag_M).imag
        abs_diag_M_sq = abs(diag_M) ** 2

        r_n = -(1 / Nk) * w_b * np.sum(log_diag_M_imag, axis=k_axes).T @ k_shell
        rsq_n = (1 / Nk) * w_b * np.sum(
            (1 - abs_diag_M_sq + log_diag_M_imag ** 2), axis=k_axes+tuple([-2]))
        spread_n = rsq_n - np.array([np.vdot(r_n[n, :], r_n[n, :]) for n in range(r_n.shape[0])])

        if decomp:
            Omega_i = w_b * n_states * k_shell.shape[0] - (1 / Nk) * w_b * np.sum(abs(M) **2)
            Omega_tilde = (1 / Nk) * w_b * ( 
                np.sum((-log_diag_M_imag - k_shell @ r_n.T)**2) + 
                np.sum(abs(M)**2) - np.sum(abs_diag_M_sq)
            )
            return [spread_n, Omega_i, Omega_tilde], r_n, rsq_n

        else:
            return spread_n, r_n, rsq_n
        

    def _get_Omega_til(self, M, w_b, k_shell):
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


    def _get_Omega_I(self, M, w_b, k_shell):
        Nk = np.prod(M.shape[:-3])
        n_states = M.shape[3]
        Omega_i = w_b * n_states * k_shell.shape[0] - (1 / Nk) * w_b * np.sum(abs(M) **2)
        return Omega_i
    
     ####### Maximally Localized WF ############

    def find_optimal_subspace(
        self, outer_states, inner_states, iter_num=100, verbose=False, tol=1e-10, alpha=1
    ):
        nks = self._nks 
        Nk = np.prod(nks)
        n_orb = self.Lattice._n_orb
        n_states = self.tilde_states._n_states
        dim_subspace = n_states
    
        # Assumes only one shell for now
        w_b, _, idx_shell = self.K_mesh.get_weights(N_sh=1)
        num_nnbrs = len(idx_shell[0])
        bc_phase = self.tilde_states.get_boundary_phase(idx_shell=idx_shell)

        P = np.einsum("...ni, ...nj -> ...ij", inner_states, inner_states.conj())

        # Projector on initial subspace at each k (for pbc of neighboring spaces)
        P_nbr = np.zeros((*nks, num_nnbrs, n_orb, n_orb), dtype=complex)
        Q_nbr = np.zeros((*nks, num_nnbrs, n_orb, n_orb), dtype=complex)
        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)

        for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
            states_pbc = np.roll(inner_states, shift=tuple(-idx_vec), axis=(0,1)) * bc_phase[..., idx, np.newaxis,  :]
            P_nbr[..., idx, :, :] = np.einsum(
                    "...ni, ...nj -> ...ij", states_pbc, states_pbc.conj()
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
            
            if abs(omega_I_prev - omega_I_new) * (iter_num - i) <= tol:
                # assuming the change in omega_i monatonically decreases, omega_i will not change
                # more than tolerance with remaining steps
                print("Omega_I has converged within tolerance. Breaking loop")
                return states_min

            if verbose:
                print(f"{i} Omega_I: {omega_I_new.real}")

            omega_I_prev = omega_I_new

        return states_min


    def mat_exp(self, M):
        eigvals, eigvecs = np.linalg.eig(M)
        U = eigvecs
        U_inv = np.linalg.inv(U)
        # Diagonal matrix of the exponentials of the eigenvalues
        exp_diagM = np.exp(eigvals)
        # Construct the matrix exponential
        expM = np.einsum('...ij, ...jk -> ...ik', U, np.multiply(U_inv, exp_diagM[..., :, np.newaxis]))
        return expM
    
    
    def find_min_unitary(self, eps=1 / 160, iter_num=10, verbose=False, tol=1e-3, grad_min=1e-3):
        """
        Finds the unitary that minimizing the gauge dependent part of the spread. 

        Args:
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
        u_wfs = self.tilde_states._u_wfs
        M = self.tilde_states._M
        w_b, k_shell, idx_shell = self.K_mesh.get_weights()
        # Assumes only one shell for now
        w_b, k_shell, idx_shell = w_b[0], k_shell[0], idx_shell[0]
        nks = self._nks
        Nk = np.prod(nks)
        num_state = self.tilde_states._n_states

        U = np.zeros((*nks, num_state, num_state), dtype=complex)  # unitary transformation
        U[...] = np.eye(num_state, dtype=complex)  # initialize as identity
        M0 = np.copy(M)  # initial overlap matrix
        M = np.copy(M)  # new overlap matrix

        # initializing
        omega_tilde_prev = self._get_Omega_til(M, w_b, k_shell)
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
            U = np.einsum("...ij, ...jk -> ...ik", U, self.mat_exp(eta * eps * G))

            for idx, idx_vec in enumerate(idx_shell):
                M[..., idx, :, :] = (
                    np.transpose(U, axes=(0,1,3,2)).conj()[..., :, :] @  M0[..., idx, :, :] 
                                    @ np.roll(U, shift=tuple(-idx_vec), axis=(0,1))[..., :, :]
                                    )

            grad_mag = np.linalg.norm(np.sum(G, axis=(0,1)))
            omega_tilde_new = self._get_Omega_til(M, w_b, k_shell)

            if abs(grad_mag) <= grad_min and abs(omega_tilde_prev - omega_tilde_new) * (iter_num - i) <= tol:
                print("Omega_tilde minimization has converged within tolerance. Breaking the loop")
                print(
                f"{i} Omega_til = {omega_tilde_new.real}, Grad mag: {grad_mag}"
                )
                u_max_loc = np.einsum('...ji, ...jm -> ...im', U, u_wfs)
                return u_max_loc, U
            
            if grad_mag_prev < grad_mag and i!=0:
                print("Warning: Gradient increasing.")

            if verbose:
                print(
                    f"{i} Omega_til = {omega_tilde_new.real}, Grad mag: {grad_mag}"
                )
            grad_mag_prev = grad_mag
            omega_tilde_prev = omega_tilde_new


        u_max_loc = np.einsum('...ji, ...jm -> ...im', U, u_wfs)
        return u_max_loc, U

    def max_loc(
        self,
        outer_state_idxs="occupied",
        iter_num_omega_i=1000,
        iter_num_omega_til=1000,
        eps=1e-3,
        tol_omega_i=1e-5,
        tol_omega_til=1e-10,
        grad_min=1e-3,
        alpha=1,
        verbose=False,
        save=False, save_name=''
    ):
        """
        Find the maximally localized Wannier functions using the projection method.

        Args:
            outer_states_idxs(list | str): Band indices for the disentanglement manifold. If "occupied", 
                will use the occupied manifold. Defaults to "occupied".
            verbose(bool): Whether to print spread during minimization.

        """

        # slicing the disentanglement manifold
        if outer_state_idxs == "occupied":
            n_occ = int(self.energy_eigstates._n_states / 2)  # assuming half filled
            outer_state_idxs = list(range(n_occ))

        outer_states = self.energy_eigstates._u_wfs[..., outer_state_idxs, :]

        # Minimizing Omega_I via disentanglement
        util_min_Wan = self.find_optimal_subspace(
            outer_states,
            self.tilde_states._u_wfs,
            iter_num=iter_num_omega_i,
            verbose=verbose, alpha=alpha, tol=tol_omega_i
        )
        self.tilde_states.set_wfs(util_min_Wan)

        # Second projection
        psi_til_til_min = self.get_psi_tilde(
            self.tilde_states._psi_wfs, self.trial_wfs, state_idx=list(range(self.tilde_states._psi_wfs.shape[2]))
        )
        self.tilde_states.set_wfs(psi_til_til_min, cell_periodic=False)

        # Finding optimal gauge
        u_max_loc, _ = self.find_min_unitary(
            eps=eps, iter_num=iter_num_omega_til, verbose=verbose, tol=tol_omega_til, grad_min=grad_min)
        self.tilde_states.set_wfs(u_max_loc)

        # Fourier transform Bloch-like states
        psi_wfs = self.tilde_states._psi_wfs
        dim_k = len(psi_wfs.shape[:-2])
        # DFT
        self.WFs = np.fft.ifftn(psi_wfs, axes=[i for i in range(dim_k)], norm=None)

        spread = self.spread_recip(decomp=True)
        self.spread = spread[0][0]
        self.omega_i = spread[0][1]
        self.omega_til = spread[0][2]
        self.centers = spread[1]

        # if save:
        #     sv_dir = 'data'
        #     if not os.path.exists(sv_dir):
        #         os.makedirs(sv_dir)
        #     sv_prefix = 'W0_max_loc'
        #     np.save(f"{sv_dir}/{sv_prefix}_{save_name}", w0)
        #     sv_prefix = 'W0_max_loc_cntrs'
        #     np.save(f"{sv_dir}/{sv_prefix}_{save_name}", expc_r)
        #     sv_prefix = 'u_wfs_max_loc'
        #     np.save(f"{sv_dir}/{sv_prefix}_{save_name}", u_max_loc)

        # ret_pckg = [w0]
        # if return_uwfs:
        #     ret_pckg.append(u_max_loc)
        # if return_wf_centers:
        #     ret_pckg.append(expc_r)
        # if return_spread:
        #     ret_pckg.append(spread)
        # return ret_pckg

    def report(self):
        assert hasattr(self, 'WFs'), "First need to set Wannier functions"
        print("Wannier function report:")
        print(" --------------- ")
        print(rf"Quadratic spread = {self.spread}")
        print(rf"Omega_i = {self.omega_i}")
        print(rf"Omega_tilde = {self.omega_til}")
        print(f"Wannier centers = {self.centers}")


    def plot(
        self, Wan_idx, plot_phase=False, plot_decay=False,
        title=None, save_name=None, omit_site=None,
        fit_deg=None, fit_rng=None, ylim=None, show=False):

        orbs = self.Lattice._orbs
        lat_vecs = self.Lattice._lat_vecs
        w0 = self.WFs
    
        nx, ny = w0.shape[0], w0.shape[1]

        supercell = [(i,j) for i in range(-int((nx-nx%2)/2), int((nx-nx%2)/2)) 
                    for j in range(-int((ny-ny%2)/2), int((ny-ny%2)/2))]

        xs = []
        ys = []
        r = []
        w0i_wt = []
        w0i_phase = []

        xs_omit = []
        ys_omit = []
        r_omit = []
        w0omit_wt = []

        r_ev = []
        r_odd = []
        w0ev_wt = []
        w0odd_wt = []

        for tx, ty in supercell:
            for i, orb in enumerate(orbs):
                phase = np.arctan(w0[tx, ty, Wan_idx, i].imag/w0[tx, ty, Wan_idx, i].real) 
                wt = np.abs(w0[tx, ty, Wan_idx, i])**2
                pos = orb[0]*lat_vecs[0] + tx*lat_vecs[0] + orb[1]*lat_vecs[1]+ ty*lat_vecs[1]

                x, y, rad = pos[0], pos[1], np.sqrt(pos[0]**2 + pos[1]**2)

                xs.append(x)
                ys.append(y)
                r.append(rad)
                w0i_wt.append(wt)
                w0i_phase.append(phase)

                if omit_site is not None and i == omit_site:
                    xs_omit.append(x)
                    ys_omit.append(y)
                    r_omit.append(rad)
                    w0omit_wt.append(wt)
                elif i%2 ==0:
                    r_ev.append(rad)
                    w0ev_wt.append(wt)
                else:
                    r_odd.append(rad)
                    w0odd_wt.append(wt)

        # numpify
        xs = np.array(xs)
        ys = np.array(ys)
        r = np.array(r)
        w0i_wt = np.array(w0i_wt)
        xs_omit = np.array(xs_omit)
        ys_omit = np.array(ys_omit)
        r_omit = np.array(r_omit)
        w0omit_wt = np.array(w0omit_wt)
        r_ev = np.array(r_ev)
        w0ev_wt = np.array(w0ev_wt)
        r_odd = np.array(r_odd)
        w0odd_wt = np.array(w0odd_wt)

        figs = []
        axs = []

        fig, ax = plt.subplots()
        figs.append(fig)
        axs.append(ax)

        # Weight plot
        scat = ax.scatter(xs, ys, c=w0i_wt, cmap='plasma', norm=LogNorm(vmin=2e-16, vmax=1))

        if omit_site is not None :
            ax.scatter(xs_omit, ys_omit, s=2, marker='x', c='g')

        cbar = plt.colorbar(scat, ax=ax)
        # cbar.set_label(rf"$|\langle \phi_{{\vec{{R}}, j}}| w_{{0, {Wan_idx}}}\rangle|^2$", rotation=270)
        cbar.set_label(rf"$|w_{Wan_idx}(\mathbf{{r}})|^2$", rotation=270)
        cbar.ax.get_yaxis().labelpad = 20
        ax.set_title(title)

        # Saving
        if save_name is not None:
            plt.savefig(f'Wan_wt_{save_name}.png')

        if show:
            plt.show()

        if plot_phase:
            # Phase plot
            fig2, ax2 = plt.subplots()
            figs.append(fig2)
            axs.append(ax2)

            scat = ax2.scatter(xs, ys, c=w0i_phase, cmap='hsv')

            cbar = plt.colorbar(scat, ax=ax2)
            cbar.set_label(
                rf"$\phi = \tan^{{-1}}(\mathrm{{Im}}[w_{{0, {Wan_idx}}}(r)]\  / \ \mathrm{{Re}}[w_{{0, {Wan_idx}}}(r)])$", 
                rotation=270)
            cbar.ax.get_yaxis().labelpad = 20
            ax2.set_title(title)

            # Saving
            if save_name is not None:
                plt.savefig(f'Wan_wt_{save_name}.png')
            
            if show:
                plt.show()

        if plot_decay:
            fig3, ax3 = plt.subplots()
            figs.append(fig3)
            axs.append(ax3)

            # binning data
            max_r = np.amax(r)
            num_bins = int(np.ceil(max_r))
            r_bins = [[i, i + 1] for i in range(num_bins)]
            r_ledge = [i for i in range(num_bins)]
            r_cntr = [0.5 + i for i in range(num_bins)]
            w0i_wt_bins = [[] for i in range(num_bins)]

            # bins of weights
            for i in range(r.shape[0]):
                for j, r_intvl in enumerate(r_bins):
                    if r_intvl[0] <= r[i] < r_intvl[1]:
                        w0i_wt_bins[j].append(w0i_wt[i])
                        break

            # average value of bins
            avg_w0i_wt_bins = []
            for i in range(num_bins):
                if len(w0i_wt_bins[i]) != 0:
                    avg_w0i_wt_bins.append(sum(w0i_wt_bins[i])/len(w0i_wt_bins[i]))

            # numpify
            avg_w0i_wt_bins = np.array(avg_w0i_wt_bins)
            r_ledge = np.array(r_ledge)
            r_cntr = np.array(r_cntr)

            if fit_rng is None:
                cutoff = int(0.7*max_r)
                init_r = int(0.2*max_r)
                fit_rng = [init_r, cutoff]
            else:
                cutoff = fit_rng[-1]

            # scatter plot
            # plt.scatter(r, w0i_wt, zorder=1, s=10)
            if omit_site is not None:
                ax3.scatter(r_omit[r_omit<cutoff], w0omit_wt[r_omit<cutoff], zorder=1, s=10, c='g', label='omitted site')

            ax3.scatter(r_ev[r_ev<cutoff], w0ev_wt[r_ev<cutoff], zorder=1, s=10, c='b', label='low energy sites')
            ax3.scatter(r_odd[r_odd<cutoff], w0odd_wt[r_odd<cutoff], zorder=1, s=10, c='r', label='high energy sites')

            # bar of avgs
            ax3.bar(r_ledge[r_ledge<cutoff], avg_w0i_wt_bins[r_ledge<cutoff], width=1, align='edge', ec='k', zorder=0, ls='-', alpha=0.3)

            # fit line
            if fit_deg is None:
                deg = 1 # polynomial fit degree
            
            r_fit = r_cntr[np.logical_and(r_cntr > fit_rng[0], r_cntr < fit_rng[1])]
            w0i_wt_fit = avg_w0i_wt_bins[np.logical_and(r_cntr > fit_rng[0], r_cntr < fit_rng[1])]
            fit = np.polyfit(r_fit, np.log(w0i_wt_fit), deg)
            fit_line = np.sum(np.array([r_fit**(deg-i) * fit[i] for i in range(deg+1)]), axis=0)
            fit_label = rf"$Ce^{{{fit[-2]: 0.2f} r  {'+'.join([fr'{c: .2f} r^{deg-j}' for j, c in enumerate(fit[:-3])])}}}$"
            ax3.plot(r_fit, np.exp(fit_line), c='lime', ls='--', lw=2.5, label=fit_label)

            ax3.legend()
            ax3.set_xlabel(r'$|\mathbf{r}|$')
            ax3.set_ylabel(rf"$|w_{Wan_idx}(\mathbf{{r}})|^2$")
            # ax.set_xlabel(r'$|\vec{R}+\vec{\tau}_j|$')
            # ax.set_xlim(-4e-1, cutoff)
            if ylim is None:
                ax3.set_ylim(0.8*min(w0i_wt[r<cutoff]), 1.5)
            else:
                ax3.set_ylim(ylim)
            ax3.set_yscale('log')

            ax3.set_title(title)

            if save_name is not None:
                plt.savefig(f'Wan_decay_{save_name}.png')

            if show:
                plt.show()
        
        return figs, axs
