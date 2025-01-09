import numpy as np
from itertools import product
from itertools import combinations_with_replacement as comb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pythtb import tb_model, wf_array
import copy


class Model(tb_model):
    def __init__(self, dim_k, dim_r, lat=None, orb=None, per=None, nspin=1):
        super().__init__(dim_k, dim_r, lat=lat, orb=orb, per=per, nspin=nspin)

        self._n_orb = super().get_num_orbitals()
        self._lat_vecs = self.get_lat_vecs()
        self._orb_vecs = self.get_orb_vecs(Cartesian=False)
        self._recip_lat_vecs = self.get_recip_lat_vecs()

    def report_geom(self):
        """Prints information about the lattice attributes."""
        print("Lattice vectors (Cartesian coordinates)")
        for idx, lat_vec in enumerate(self._lat_vecs):
            print(f"a_{idx} ===> {lat_vec}")
        print("Reciprocal lattice vectors (1/Cartesian coordinates)")
        for idx, recip_lat_vec in enumerate(self._recip_lat_vecs):
            print(f"b_{idx} ===> {recip_lat_vec}")
        orb_pos = self._orb_vecs @ self._lat_vecs
        print("Position of orbitals (Cartesian)")
        for idx, pos in enumerate(orb_pos):
            print(f"{idx} ===> {pos}")
        print("Position of orbitals (reduced coordinates)")
        for idx, pos in enumerate(self._orb_vecs):
            print(f"{idx} ===> {pos}")
    

    def get_lat_vecs(self):
        return super().get_lat()

    def get_orb_vecs(self, Cartesian: bool = False):
        """Returns orbtial vectors."""
        orb_vecs = super().get_orb()
        if Cartesian:
            return orb_vecs @ self._lat_vecs
        else:
            return orb_vecs
        
    def get_recip_lat_vecs(self):
        b = 2 * np.pi * np.linalg.inv(self._lat_vecs).T
        return b
    
    def get_recip_vol(self):
        return abs(np.linalg.det(self._recip_lat_vecs))

    def get_ham(self, k_pts=None):
        """Generate Bloch Hamiltonian for an array of k-points.

        K-point is given in reduced coordinates!"""
        
        k_arr = np.array(k_pts)
        if k_pts is not None:
            # if kpnt is just a number then convert it to an array
            if len(k_arr.shape) == 0:
                k_arr = np.array([k_arr])
                n_kpts = 1
            else:
                n_kpts = k_arr.shape[0]
            # check that k-vector is of corect size
            if k_arr.shape[-1] != self._dim_k:
                raise Exception("\n\nk-vector of wrong shape!")
        else:
            if self._dim_k != 0:
                raise Exception("\n\nHave to provide a k-vector!")

        if self._nspin == 1:
            if n_kpts == 1:
                ham = np.zeros((self._norb, self._norb), dtype=complex)
            else:
                ham = np.zeros((n_kpts, self._norb, self._norb), dtype=complex)
        elif self._nspin == 2:
            if n_kpts == 1:
                ham = np.zeros((self._norb, 2, self._norb, 2), dtype=complex)
            else:
                ham = np.zeros((n_kpts, self._norb, 2, self._norb, 2), dtype=complex)
        else:
            raise ValueError("Invalid spin value.")
        
        # set diagonal elements
        for i in range(self._norb):
            if self._nspin == 1:
                ham[..., i, i] = self._site_energies[i]
            elif self._nspin == 2:
                ham[..., i, :, i, :] = self._site_energies[i]

        # Loop over all hoppings
        for hopping in self._hoppings:
            if self._nspin == 1:
                amp = complex(hopping[0])
            elif self._nspin == 2:
                amp = np.array(hopping[0], dtype=complex)

            i = hopping[1]
            j = hopping[2]

            if self._dim_k > 0:
                ind_R = np.array(hopping[3], dtype=float)

                # Compute delta_r for periodic directions
                delta_r = ind_R - self._orb[i, :] + self._orb[j, :]  # Shape: (dim_r,)
                delta_r_per = delta_r[self._per]  # Shape: (dim_k,)

                # Compute phase factors for all k-points
                phase = np.exp(1j * 2 * np.pi * k_arr @ delta_r_per)  # Shape: (n_kpts,)

                # Compute the amplitude for all k-points and components
                amp *= phase  # Shape: (n_kpts,)

            if self._nspin == 1:
                ham[..., i, j] += amp 
                ham[..., j, i] += amp.conjugate()
            elif self._nspin == 2:
                ham[..., i, :, j, :] += amp
                ham[..., j, :, i, :] += amp.T.conjugate()

        return ham
    
    
    def solve_ham(self, k_arr, return_eigvecs=False):
        H_k = self.get_ham(k_arr)
        eigvals, eigvecs = np.linalg.eigh(H_k)
        eigvecs = np.transpose(eigvecs, axes=(0, 2, 1))  # [k, n, orb]

        if return_eigvecs:
            return eigvals, eigvecs
        else:
            return eigvals
    

    def gen_velocity(self, k_pts):
        """
        Generate the velocity operator for an array of k-points.
        
        Parameters:
            model: Tight-binding model instance.
            k_pts: Array of k-points in reduced coordinates, shape (n_kpts, dim_k).
        
        Returns:
            vel: Velocity operators at each k-point, shape (n_kpts, dim_k, n_orb, n_orb).
        """
        
        k_pts = np.array(k_pts)
        n_kpts, dim_k = k_pts.shape

        if dim_k != self._dim_k:
            raise ValueError("k-points have incorrect dimension.")

        n_orb = self._norb

        if self._nspin == 1:
            vel = np.zeros((n_kpts, dim_k, n_orb, n_orb), dtype=complex)
        elif self._nspin == 2:
            vel = np.zeros((n_kpts, dim_k, n_orb, 2, n_orb, 2), dtype=complex)
        else:
            raise ValueError("Invalid spin value.")

        # Precompute the lattice vectors for periodic directions
        lat_per = self.get_lat()[self._per, :]  # Shape: (dim_k, dim_k)

        # Loop over all hoppings
        for hopping in self._hoppings:
            if self._nspin == 1:
                amp = complex(hopping[0])
            elif self._nspin == 2:
                amp = np.array(hopping[0], dtype=complex)

            i = hopping[1]
            j = hopping[2]
            ind_R = np.array(hopping[3], dtype=float)

            # Compute delta_r for periodic directions
            delta_r = ind_R - self._orb[i, :] + self._orb[j, :]  # Shape: (dim_r,)
            delta_r_per = delta_r[self._per]  # Shape: (dim_k,)

            # Convert delta_r to Cartesian coordinates
            delta_r_Cart = delta_r_per @ lat_per  # Shape: (dim_k,)

            # Compute phase factors for all k-points
            phase = np.exp(1j * 2 * np.pi * k_pts @ delta_r_per)  # Shape: (n_kpts,)

            # Compute the amplitude for all k-points and components
            delta_amp = (-1j) * (-delta_r_Cart)  # Shape: (dim_k,)
            amp_k = amp * phase[:, np.newaxis] * delta_amp[np.newaxis, :]  # Shape: (n_kpts, dim_k)

            if self._nspin == 1:
                # Update velocity operator
                vel[..., :, i, j] += amp_k 
                vel[..., :, j, i] += np.conj(amp_k)
            elif self._nspin == 2:
                vel[..., :, i, :, j, :] += amp
                vel[..., :, j, :, i, :] += amp.T.conjugate()

        return vel
    

    def quantum_geom_tens(self, k_pts, occ_idxs=None):
        dim_k = k_pts.shape[-1]
        Nk = k_pts.shape[0]
        
        evals, evecs = self.solve_ham(k_pts, return_eigvecs=True)
        n_eigs = evecs.shape[1]

        if occ_idxs is None:
            occ_idxs = list(range(int(n_eigs/2)))

        v_k = self.gen_velocity(k_pts)
        v_k_rot = np.einsum("...ni, ...aij, ...mj -> ...anm", evecs.conj(), v_k, evecs)  # (n_kpts, dim_k, n_orb, n_orb)
        delta_E = evals[..., np.newaxis, :] - evals[..., :, np.newaxis]
        delta_E_sq = delta_E**2
        QGT = np.zeros((Nk, n_eigs, dim_k, dim_k), dtype=complex)

        for n in range(n_eigs):
            for m in range(n_eigs):
                if (n in occ_idxs) ^ (m in occ_idxs):
                    for mu in range(dim_k):
                        v_nm_mu = v_k_rot[:, mu, n, m]
                        for nu in range(dim_k):
                            v_mn_nu = v_k_rot[:, nu, m, n]
                            QGT[:, n, mu, nu] += (v_nm_mu * v_mn_nu) / delta_E_sq[:, n, m]

        return QGT
    
    
    def Chern_QGT(self, Omega=None):
        if Omega is None:
            nkx, nky = 50, 50
            k_mesh = K_mesh(self, nkx, nky)
            full_mesh = k_mesh.gen_k_mesh(endpoint=False)

            QGT = self.quantum_geom_tens(full_mesh)
            Omega = -2 * QGT.imag 

        Nk = Omega.shape[0] 
        V_BZ = self.get_recip_vol()
        dk_sq = V_BZ / Nk
        Chern = np.sum(Omega[..., 0, 1], axis=0) * dk_sq / (2 * np.pi)

        return Chern
        
    
    def make_supercell(
            self, sc_red_lat, return_sc_vectors=False, to_home=True, to_home_suppress_warning=False
            ):
        
        # Can't make super cell for model without periodic directions
        if self._dim_r==0:
            raise Exception("\n\nMust have at least one periodic direction to make a super-cell")
        
        # convert array to numpy array
        use_sc_red_lat=np.array(sc_red_lat)
        
        # checks on super-lattice array
        if use_sc_red_lat.shape != (self._dim_r, self._dim_r):
            raise Exception("\n\nDimension of sc_red_lat array must be dim_r*dim_r")
        if use_sc_red_lat.dtype!=int:
            raise Exception("\n\nsc_red_lat array elements must be integers")
        for i in range(self._dim_r):
            for j in range(self._dim_r):
                if (i==j) and (i not in self._per) and use_sc_red_lat[i,j] != 1:
                    raise Exception("\n\nDiagonal elements of sc_red_lat for non-periodic directions must equal 1.")
                if (i!=j) and ((i not in self._per) or (j not in self._per)) and use_sc_red_lat[i,j]!=0:
                    raise Exception("\n\nOff-diagonal elements of sc_red_lat for non-periodic directions must equal 0.")
        if np.abs(np.linalg.det(use_sc_red_lat))<1.0E-6:
            raise Exception("\n\nSuper-cell lattice vectors length/area/volume too close to zero, or zero.")
        if np.linalg.det(use_sc_red_lat)<0.0:
            raise Exception("\n\nSuper-cell lattice vectors need to form right handed system.")

        # converts reduced vector in original lattice to reduced vector in super-cell lattice
        def to_red_sc(red_vec_orig):
            return np.linalg.solve(np.array(use_sc_red_lat.T,dtype=float),
                                   np.array(red_vec_orig,dtype=float))

        # conservative estimate on range of search for super-cell vectors
        max_R = np.max(np.abs(use_sc_red_lat))*self._dim_r

        # candidates for super-cell vectors
        # CHANGED
        sc_cands = [
            np.array(tup) 
            for tup in product(*[range(-max_R, max_R + 1) for i in range(self._dim_r)])
            ]
        if self._dim_r < 1 or self._dim_r > 4:
            raise Exception("\n\nWrong dimensionality of dim_r!")

        # find all vectors inside super-cell
        # store them here
        sc_vec = []
        eps_shift = np.sqrt(2.0)*1.0E-8 # shift of the grid, so to avoid double counting
        for vec in sc_cands:
            # compute reduced coordinates of this candidate vector in the super-cell frame
            tmp_red=to_red_sc(vec).tolist()
            # check if in the interior
            inside=True
            for t in tmp_red:
                if t<=-1.0*eps_shift or t>1.0-eps_shift:
                    inside=False                
            if inside:
                sc_vec.append(np.array(vec))
        # number of times unit cell is repeated in the super-cell
        num_sc=len(sc_vec)

        # check that found enough super-cell vectors
        if int(round(np.abs(np.linalg.det(use_sc_red_lat))))!=num_sc:
            raise Exception("\n\nSuper-cell generation failed! Wrong number of super-cell vectors found.")

        # cartesian vectors of the super lattice
        sc_cart_lat=np.dot(use_sc_red_lat,self._lat)

        # orbitals of the super-cell tight-binding model
        # CHANGED
        sc_orb = [ to_red_sc(orb+cur_sc_vec) 
            for cur_sc_vec in sc_vec for orb in self._orb
            ]
        
        # create super-cell tb_model object to be returned
        # CHANGED
        sc_tb = Model(self._dim_k,self._dim_r,sc_cart_lat,sc_orb,per=self._per,nspin=self._nspin)

        # remember if came from w90
        sc_tb._assume_position_operator_diagonal=self._assume_position_operator_diagonal

        # repeat onsite energies
        for i in range(num_sc):
            for j in range(self._norb):
                sc_tb.set_onsite(self._site_energies[j],i*self._norb+j)

        # set hopping terms
        for c,cur_sc_vec in enumerate(sc_vec): # go over all super-cell vectors
            for h in range(len(self._hoppings)): # go over all hopping terms of the original model
                # amplitude of the hop is the same
                amp=self._hoppings[h][0]

                # lattice vector of the hopping
                ind_R=copy.deepcopy(self._hoppings[h][3])
                # super-cell component of hopping lattice vector
                # shift also by current super cell vector
                sc_part=np.floor(to_red_sc(ind_R+cur_sc_vec)) # round down!
                sc_part=np.array(sc_part,dtype=int)
                # find remaining vector in the original reduced coordinates
                orig_part=ind_R+cur_sc_vec-np.dot(sc_part,use_sc_red_lat)
                # remaining vector must equal one of the super-cell vectors
                pair_ind=None
                for p,pair_sc_vec in enumerate(sc_vec):
                    if False not in (pair_sc_vec==orig_part):
                        if pair_ind is not None:
                            raise Exception("\n\nFound duplicate super cell vector!")
                        pair_ind=p
                if pair_ind is None:
                    raise Exception("\n\nDid not find super cell vector!")
                        
                # index of "from" and "to" hopping indices
                hi=self._hoppings[h][1] + c*self._norb
                hj=self._hoppings[h][2] + pair_ind*self._norb
                
                # add hopping term
                sc_tb.set_hop(amp,hi,hj,sc_part,mode="add",allow_conjugate_pair=True)

        # put orbitals to home cell if asked for
        if to_home:
            sc_tb._shift_to_home(to_home_suppress_warning)

        # return new tb model and vectors if needed
        if not return_sc_vectors:
            return sc_tb
        else:
            return (sc_tb,sc_vec)
        
        
    def plot_bands(
        self, k_path, nk=101, k_label=None, red_lat_idx=None, 
        fig=None, ax=None, title=None, scat_size=3, 
        lw=3, lc='b', ls='solid', cmap="bwr", show=False, label=None, cbar=True
        ):
        """

        Args:
            k_path (list): List of high symmetry points to plot bands through
            k_label (list[str], optional): Labels of high symmetry points. Defaults to None.
            title (str, optional): _description_. Defaults to None.
            save_name (str, optional): _description_. Defaults to None.
            red_lat_idx (list, optional): _description_. Defaults to None.
            show (bool, optional): _description_. Defaults to False.

        Returns:
            fig, ax: matplotlib fig and ax
        """
        
        if fig is None:
            fig, ax = plt.subplots()

        # generate k-path and labels
        (k_vec, k_dist, k_node) = self.k_path(k_path, nk, report=False)
        # diagonalize model on path
        evals, evecs = self.solve_ham(k_vec, return_eigvecs=True)
        n_eigs = evecs.shape[-2]

        # scattered bands with sublattice color
        if red_lat_idx is not None:
            wt = abs(evecs)**2
            col = np.sum([  wt[..., i] for i in red_lat_idx ], axis=0)

            for n in range(n_eigs):
                scat = ax.scatter(k_dist, evals[:, n], c=col[:, n], cmap=cmap, marker='o', s=scat_size, vmin=0, vmax=1, zorder=2)

            if cbar:
                cbar = fig.colorbar(scat, ticks=[1,0])
                cbar.ax.set_yticklabels([r'$\psi_2$', r'$\psi_1$'], size=12)

        else:
            # continuous bands
            for n in range(n_eigs):
                ax.plot(k_dist, evals[:, n], c=lc, lw=lw, ls=ls, label=label)

        ax.set_xlim(0, k_node[-1])
        ax.set_xticks(k_node)
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n], linewidth=0.5, color='k', zorder=1)
        if k_label is not None:
            ax.set_xticklabels(k_label, size=12)
        
        ax.set_title(title)
        ax.set_ylabel(r"Energy $E(\mathbf{{k}})$", size=12)
        ax.yaxis.labelpad = 10

        if show:
            plt.show()

        return fig, ax


class K_mesh():
    def __init__(self, model: Model, *nks):
        """Class for storing and manipulating a regular mesh of k-points. 
        """
        self.model = model
        self.nks = nks
        self.dim: int = len(nks)
        self.idx_arr: list = list(product(*[range(nk) for nk in nks]))  # 1D list of all k_indices (integers)
        self.full_mesh: np.ndarray = self.gen_k_mesh(flat=False, endpoint=False) # each index is a direction in k-space
        self.flat_mesh: np.ndarray = self.gen_k_mesh(flat=True, endpoint=False) # 1D list of k-vectors

        # nearest neighbor k-shell
        self.nnbr_w_b, _, self.nnbr_idx_shell = self.get_weights(N_sh=1)
        self.num_nnbrs = len(self.nnbr_idx_shell[0])

        # matrix of e^{-i G . r} phases
        self.bc_phase = self.get_boundary_phase()
        self.orb_phases = self.get_orb_phases()

    def gen_k_mesh(
            self, 
            centered: bool = False, 
            flat: bool = True, 
            endpoint: bool = False
            ) -> np.ndarray:
        """Generate a regular k-mesh in reduced coordinates. 

        Args:
            centered (bool): 
                If True, mesh is defined from [-0.5, 0.5] along each direction. 
                Defaults to False.
            flat (bool):
                If True, returns flattened array of k-points (e.g. of dimension nkx*nky*nkz x 3). 
                If False, returns reshaped array with axes along each k-space dimension 
                (e.g. of dimension nkx x nky x nkz x 3). Defaults to True.
            endpoint (bool): 
                If True, includes 1 (edge of BZ in reduced coordinates) in the mesh. Defaults to False. When Wannierizing shoule 

        Returns:
            k-mesh (np.ndarray): 
                Array of k-mesh coordinates.
        """
        end_pts = [-0.5, 0.5] if centered else [0, 1]

        k_vals = [np.linspace(end_pts[0], end_pts[1], nk, endpoint=endpoint) for nk in self.nks]
        mesh = np.array(list(product(*k_vals)))

        return mesh if flat else mesh.reshape(*[nk for nk in self.nks], len(self.nks))
    
    def get_k_shell(
            self, 
            N_sh: int, 
            report: bool = False
            ):
        """Generates shells of k-points around the Gamma point.

        Returns array of vectors connecting the origin to nearest neighboring k-points 
        in the mesh, along with vectors of reduced coordinates. 

        Args:
            N_sh (int): 
                Number of nearest neighbor shells.
            report (bool):
                If True, prints a summary of the k-shell.

        Returns:
            k_shell (np.ndarray[float]):
                Array of vectors in inverse units of lattice vectorsconnecting nearest neighbor k-mesh points.
            idx_shell (np.ndarray[int]):
                Array of vectors of integers used for indexing the nearest neighboring k-mesh points
                to a given k-mesh point.
        """
        # basis vectors connecting neighboring mesh points (in inverse Cartesian units)
        dk = np.array([self.model._recip_lat_vecs[i] / nk for i, nk in enumerate(self.nks)])
        # array of integers e.g. in 2D for N_sh = 1 would be [0,1], [1,0], [0,-1], [-1,0]
        nnbr_idx = list(product(list(range(-N_sh, N_sh + 1)), repeat=len(self.nks)))
        nnbr_idx.remove((0, 0))
        nnbr_idx = np.array(nnbr_idx)
        # vectors connecting k-points near Gamma point (in inverse lattice vector units)
        b_vecs = np.array([nnbr_idx[i] @ dk for i in range(nnbr_idx.shape[0])])
        # distances to points around Gamma
        dists = np.array([np.vdot(b_vecs[i], b_vecs[i]) for i in range(b_vecs.shape[0])])
        # remove numerical noise
        dists = dists.round(10)

        # sorting by distance
        sorted_idxs = np.argsort(dists)
        dists_sorted = dists[sorted_idxs]
        b_vecs_sorted = b_vecs[sorted_idxs]
        nnbr_idx_sorted = nnbr_idx[sorted_idxs]

        unique_dists = sorted(list(set(dists))) # removes repeated distances
        keep_dists = unique_dists[:N_sh] # keep only distances up to N_sh away
        # keep only b_vecs in N_sh shells
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
        """Generates the finite difference weights on a k-shell.
        """
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
    
    
    def get_boundary_phase(self):
        """
        Get phase factors to multiply the cell periodic states in the first BZ
        related by the pbc u_{n, k+G} = u_{n, k} exp(-i G . r)

        Returns:
            bc_phase (np.ndarray): 
                The shape is [...k(s), shell_idx] where shell_idx is an integer
                corresponding to a particular idx_vec where the convention is to go  
                counter-clockwise (e.g. square lattice 0 --> [1, 0], 1 --> [0, 1] etc.)

        """
        idx_shell = self.nnbr_idx_shell
        bc_phase = np.ones((*self.nks, idx_shell[0].shape[0], self.model._orb_vecs.shape[0]), dtype=complex)
        for k_idx in self.idx_arr: # each index in k-mesh
            for shell_idx, idx_vec in enumerate(idx_shell[0]):  # vecs connecting neighboring indices
                k_nbr_idx = np.array(k_idx) + idx_vec # indices for neighboring k
                mod_idx = np.mod(k_nbr_idx, self.nks) # apply pbc to index
                diff = k_nbr_idx - mod_idx # if mod changed nbr, will be non-zero
                G = np.divide(np.array(diff), np.array(self.nks)) # will be pm 1 where crossed, 0 else
                # if the translated k-index contains -1 or nk_i+1 then we crossed the BZ boundary
                cross_bndry = np.any((k_nbr_idx == -1) | np.logical_or.reduce([k_nbr_idx == nk for nk in self.nks]))
                if cross_bndry:
                    bc_phase[k_idx][shell_idx]= np.exp(-1j * 2 * np.pi * self.model._orb_vecs @ G.T).T

        return bc_phase
    

    def get_orb_phases(self, inverse=False):
        """Returns exp(\pm i k.tau) factors

        Args:
            Inverse (bool):
                If True, multiplies factor of -1 for mutiplying Bloch states to get cell-periodic states. 
        """
        lam = -1 if inverse else 1  # overall minus if getting cell periodic from Bloch
        per_dir = list(range(self.flat_mesh.shape[-1]))  # list of periodic dimensions
        # slice second dimension to only keep only periodic dimensions in orb
        per_orb = self.model._orb_vecs[:, per_dir]

        # compute a list of phase factors [k_val, orbital]
        wf_phases = np.exp(lam * 1j * 2 * np.pi * per_orb @ self.flat_mesh.T, dtype=complex).T
        return wf_phases  # 1D numpy array of dimension norb
    
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
        phase = np.exp(-1j * 2 * np.pi * orbs @ np.array(G).T).T
        return phase


class Bloch():
    def __init__(self, model: Model, *nks):
        """Class for storing and manipulating Bloch like wavefunctions.
        
        Wavefunctions are defined on a semi-full reciprocal space mesh.
        """
        self.model: Model = model
        self.K_mesh: K_mesh = K_mesh(model, *nks)
        self.set_Bloch_ham()

    
    def solve_on_path(self, k_arr):
        eigvals, eigvecs = self.model.solve_ham(k_arr, return_eigvecs=True)
        self.set_wfs(eigvecs)
        self.energies = eigvals


    def solve_model(self):
        """
        Solves for the eigenstates of the Bloch Hamiltonian defined by the model over a semi-full 
        k-mesh, e.g. in 3D reduced coordinates {k = [kx, ky, kz] | k_i in [0, 1)}.
        """

        eigvals, eigvecs = self.model.solve_ham(self.K_mesh.flat_mesh, return_eigvecs=True)
        eigvecs = eigvecs.reshape(*[nk for nk in self.K_mesh.nks], *eigvecs.shape[1:])
        eigvals = eigvals.reshape(*[nk for nk in self.K_mesh.nks], *eigvals.shape[1:])
        self.set_wfs(eigvecs)
        self.energies = eigvals

    def get_states(self):
        """Returns dictionary containing Bloch and cell-periodic eigenstates."""
        assert hasattr(self, "_psi_wfs"), "Need to call `solve_model` or `set_wfs` to initialize Bloch states"
        return {"Bloch": self._psi_wfs, "Cell periodic": self._u_wfs}
    
    def get_projector(self, return_Q = False):
        assert hasattr(self, "_P"), "Need to call `solve_model` or `set_wfs` to initialize Bloch states"
        if return_Q:
            return self._P, self._Q
        else:
            return self._P
    
    def get_nbr_projector(self, return_Q = False):
        assert hasattr(self, "_P_nbr"), "Need to call `solve_model` or `set_wfs` to initialize Bloch states"
        if return_Q:
            return self._P_nbr, self._Q_nbr
        else:
            return self._P_nbr

    def get_energies(self):
        assert hasattr(self, "energies"), "Need to call `solve_model` to initialize energies"
        return self.energies
    
    def get_Bloch_Ham(self):
        """Returns the Bloch Hamiltonian of the model defined over the semi-full k-mesh."""
        return self.H_k
    
    def get_overlap_mat(self):
        """Returns overlap matrix.
        
        Overlap matrix defined as M_{n,m,k,b} = <u_{n, k} | u_{m, k+b}>
        """
        assert hasattr(self, "_M"), "Need to call `solve_model` or `set_wfs` to initialize overlap matrix"
        return self._M
    
    def set_Bloch_ham(self):
        H_k = self.model.get_ham(k_pts=self.K_mesh.flat_mesh)
        self.H_k = H_k.reshape(*[nk for nk in self.K_mesh.nks], *H_k.shape[1:])

    def set_wfs(self, wfs, cell_periodic: bool=True):
        """
        Sets the Bloch and cell-periodic eigenstates as class attributes.

        Args:
            wfs (np.ndarray): 
                Bloch (or cell-periodic) eigenstates defined on a semi-full k-mesh corresponding
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
        self._M = self.self_overlap_mat()
        # band projectors
        self._P = np.einsum("...ni, ...nj -> ...ij", self._u_wfs, self._u_wfs.conj())
        self._Q = np.eye(self._P.shape[-1]) - self._P[..., :, :]

        nks = self.K_mesh.nks
        n_orb = self.model._n_orb
        num_nnbrs = self.K_mesh.num_nnbrs
        nnbr_idx_shell = self.K_mesh.nnbr_idx_shell

        self._P_nbr = np.zeros((*nks, num_nnbrs, n_orb, n_orb), dtype=complex)
        self._Q_nbr = np.zeros((*nks, num_nnbrs, n_orb, n_orb), dtype=complex)
        for idx, idx_vec in enumerate(nnbr_idx_shell[0]):  # nearest neighbors
            # accounting for phase across the BZ boundary
            states_pbc = np.roll(self._u_wfs, shift=tuple(-idx_vec), axis=(0,1)) * self.K_mesh.bc_phase[..., idx, np.newaxis,  :]
            self._P_nbr[..., idx, :, :] = np.einsum(
                    "...ni, ...nj -> ...ij", states_pbc, states_pbc.conj()
                    )
            self._Q_nbr[..., idx, :, :] = np.eye(n_orb) - self._P_nbr[..., idx, :, :]

                
    def apply_phase(self, wfs, inverse=False):
        """
        Change between cell periodic and Bloch wfs by multiplying exp(\pm i k . tau)

        Args:
        wfs (pythtb.wf_array): Bloch or cell periodic wfs [k, nband, norb]

        Returns:
        wfsxphase (np.ndarray): 
            wfs with orbitals multiplied by phase factor

        """
        phases = self.K_mesh.get_orb_phases(inverse=inverse).reshape(*self.K_mesh.nks, self.model._n_orb)
    
        # Broadcasting the phases to match dimensions
        wfsxphase = wfs * phases[..., np.newaxis, :] 
        return wfsxphase
    
    
    def self_overlap_mat(self):
        """Compute the overlap matrix of the cell periodic eigenstates. 
        
        Overlap matrix of the form
        
        M_{m,n}^{k, k+b} = < u_{m, k} | u_{n, k+b} >

        Assumes that the last u_wf along each periodic direction corresponds to the
        next to last k-point in the mesh (excludes endpoints). 

        Returns:
            M (np.array): 
                Overlap matrix with shape [*nks, num_nnbrs, n_states, n_states]
        """

        # Assumes only one shell for now
        _, idx_shell = self.K_mesh.get_k_shell(N_sh=1, report=False)
        bc_phase = self.K_mesh.bc_phase

        M = np.zeros(
            (*self.K_mesh.nks, len(idx_shell[0]), self._n_states, self._n_states), dtype=complex
        )  # overlap matrix
        
        for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
            # introduce phases to states when k+b is across the BZ boundary
            states_pbc = np.roll(self._u_wfs, shift=tuple(-idx_vec), axis=(0,1)) * bc_phase[..., idx, np.newaxis,  :]
            M[..., idx, :, :] = np.einsum("...mj, ...nj -> ...mn", self._u_wfs.conj(), states_pbc)
        return M
    
    
    def overlap_mat(self, psi_wfs, tfs, state_idx):
        """
        Returns A_{k, n, j} = <psi_{n,k} | t_{j}> where psi are Bloch states and t are
        the trial wavefunctions.

        Args:
            psi_wfs (np.array): Bloch eigenstates
            tfs (np.array): trial wfs
            state_idx (list): band indices to form overlap matrix with

        Returns:
            A (np.array): overlap matrix
        """
        psi_wfs = np.take(psi_wfs, state_idx, axis=-2)
        A = np.einsum("...ij, kj -> ...ik", psi_wfs.conj(), tfs)
        return A
    

    def Berry_phase(self, dir=0):
        """
        Computes Berry phases for wavefunction arrays defined in parameter space.

        Parameters:
            wfs (np.ndarray): 
                Wavefunction array of shape [*param_arr_lens, n_orb, n_orb] where
                axis -2 corresponds to the eigenvalue index and axis -1 corresponds
                to amplitude.
            dir (int): 
                The direction (axis) in the parameter space along which to compute the Berry phase.

        Returns:
            phase (np.ndarray): 
                Berry phases for the specified parameter space direction.
        """
        wfs = self.get_states()["Cell periodic"]
        dim_param = len(wfs.shape[:-2]) # dimensionality of parameter space
        param_axes = np.arange(0, dim_param) # parameter axes
        param_axes = np.setdiff1d(param_axes, dir) # remove direction from axes to loop
        lens = [wfs.shape[ax] for ax in param_axes] # sizes of loop directions
        idxs = np.ndindex(*lens) # index mesh
        
        phase = np.zeros((*lens, wfs.shape[-2]))
        
        for idx_set in idxs:
            # take wfs along loop axis at given idex
            sliced_wf = wfs.copy()
            for ax, idx in enumerate(idx_set):
                sliced_wf = np.take(sliced_wf, idx, axis=param_axes[ax])

            # wf now has 3 indices: [phase ax, eigval idx, orb amp]
            for n in range(sliced_wf.shape[-2]): # loop over eigval idxs
                prod = np.prod(
                    [ np.vdot(sliced_wf[i, n], sliced_wf[i+1, n]) 
                    for i in range(sliced_wf.shape[0]-1) ] )
                prod *= np.vdot(sliced_wf[-1, n], sliced_wf[0, n])
                phase[idx_set][n] = -np.angle(prod)

        return phase
    

    def berry_loop(self, wf, berry_evals=False):
        """Do one Berry phase calculation (also returns a product of M
        matrices).  Always returns numbers between -pi and pi.  wf has
        format [kpnt,band,orbital,spin] and kpnt has to be one dimensional.
        Assumes that first and last k-point are the same. Therefore if
        there are n wavefunctions in total, will calculate phase along n-1
        links only!  If berry_evals is True then will compute phases for
        individual states, these corresponds to 1d hybrid Wannier
        function centers. Otherwise just return one number, Berry phase."""
        # number of occupied states
        nocc = wf.shape[1]
        # temporary matrices
        prd = np.identity(nocc, dtype=complex)
        ovr = np.zeros([nocc, nocc], dtype=complex)
        # go over all pairs of k-points, assuming that last point is overcounted!
        for i in range(wf.shape[0]-1):
            # generate overlap matrix, go over all bands
            for j in range(nocc):
                for k in range(nocc):
                    # ovr[j,k] = wf_dpr(wf[i,j,:], wf[i+1,k,:])
                    ovr[j,k] = np.vdot(wf[i,j,:], wf[i+1,k,:])

            # only find Berry phase
            if not berry_evals:
                # multiply overlap matrices
                prd = np.dot(prd, ovr)
            # also find phases of individual eigenvalues
            else:
                # cleanup matrices with SVD then take product
                matU, sing, matV = np.linalg.svd(ovr)
                prd = np.dot(prd, np.dot(matU, matV))
        # calculate Berry phase
        if not berry_evals:
            det = np.linalg.det(prd)
            pha = (-1.0)*np.angle(det)
            return pha
        # calculate phases of all eigenvalues
        else:
            evals = np.linalg.eigvals(prd)
            eval_pha = (-1.0)*np.angle(evals)
            # sort these numbers as well
            eval_pha = np.sort(eval_pha)
            return eval_pha
        

    def berry_flux_plaq(self, state_idx=None):
        "Compute fluxes on a two-dimensional plane of states."
        wfs = self.get_states()["Cell periodic"]
        orb_vecs = self.model._orb_vecs
        # size of the mesh
        nks = self.K_mesh.nks
        nk0 = wfs.shape[0]
        nk1 = wfs.shape[1]

        # here store flux through each plaquette of the mesh
        all_phases = np.zeros((nk0, nk1), dtype=float)

        if state_idx is not None:
            wfs = np.take(wfs, state_idx, axis=-2)

        # go over all plaquettes
        for i in range(nk0):
            for j in range(nk1):
                wf_use = [wfs[i, j]]

                # generate a small loop made out of four pieces
                nbrs = [[i+1, j], [i+1, j+1], [i, j+1]]
                for nbr in nbrs:
                    mod_idx = np.mod(nbr, nks) # apply pbc to index
                    wf_nbr = np.copy(wfs[tuple(mod_idx)]) # copy to prevent modifying original array

                    diff = nbr - mod_idx # if mod changed nbr, will be non-zero
                    if np.any(diff):
                        G = np.divide(np.array(diff), np.array(nks))  # will be pm 1 where crossed, 0 else
                        phase = np.exp(-2j * np.pi * G @ orb_vecs.T)
                        wf_nbr *= phase[np.newaxis, ...]
                    
                    wf_use.append(wf_nbr)
                
                wf_use.append(wfs[i, j])
                wf_use = np.array(wf_use, dtype=complex)
                # calculate phase around one plaquette
                all_phases[i, j] = self.berry_loop(wf_use)

        return all_phases
    

    def trace_metric(self):
        P = self._P
        Q_nbr = self._Q_nbr

        nks = Q_nbr.shape[:-3]
        num_nnbrs = Q_nbr.shape[-3]
        w_b, _, _ = self.K_mesh.get_weights(N_sh=1)

        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for nbr_idx in range(num_nnbrs):  # nearest neighbors
            T_kb[..., nbr_idx] = np.trace(P[..., :, :] @ Q_nbr[..., nbr_idx, :, :], axis1=-1, axis2=-2)

        return w_b[0] * np.sum(T_kb, axis=-1)
    
    def omega_til(self):
        M = self._M
        w_b, k_shell, idx_shell = self.K_mesh.get_weights(N_sh=1)
        w_b = w_b[0]
        k_shell = k_shell[0]

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
    

    def Chern_num(self):
        n_occ = int(self.model._n_orb/2)
        berry_curv = self.berry_flux_plaq(state_idx=np.arange(n_occ))
        Chern = np.sum(berry_curv/(2*np.pi))
        return Chern
    

    def interp_op(self, O_k, k_path, plaq=False):
        k_mesh = np.copy(self.K_mesh.full_mesh)
        k_idx_arr = self.K_mesh.idx_arr
        nks = self.K_mesh.nks
        Nk = np.prod([nks])

        if plaq:
            k_mesh += np.array([(1/nk)/2 for nk in nks])

        # Fourier transform to real space
        supercell = list(product(*[range(-int((nk-nk%2)/2), int((nk-nk%2)/2)+1) for nk in nks]))
        O_R = np.zeros((len(supercell), *O_k.shape[2:]), dtype=complex)
        for idx, (x, y) in enumerate(supercell):
            for k_idx in k_idx_arr:
                R_vec = np.array([x, y])
                phase = np.exp(-1j * 2 * np.pi * np.vdot(k_mesh[k_idx], R_vec))
                O_R[idx] += O_k[k_idx] * phase / Nk

        # interpolate to arbitrary k
        O_k_interp = np.zeros((k_path.shape[0], *O_R.shape[2:]), dtype=complex)
        for k_idx, k in enumerate(k_path):
            for idx, (x, y) in enumerate(supercell):
                R_vec = np.array([x, y])
                phase = np.exp(1j * 2 * np.pi * np.vdot(k, R_vec))
                O_k_interp[k_idx] += O_R[idx] * phase

        return O_k_interp
    
    
    def plot_bands(
        self, k_path, 
        nk=101, k_label=None, title=None, scat_size=3, lw=3, lc='b', 
        save_name=None, red_lat_idx=None, 
        fig=None, ax=None, show=False
        ):
        """

        Args:
            k_path (list): List of high symmetry points to plot bands through
            k_label (list[str], optional): Labels of high symmetry points. Defaults to None.
            title (str, optional): _description_. Defaults to None.
            save_name (str, optional): _description_. Defaults to None.
            red_lat_idx (list, optional): _description_. Defaults to None.
            show (bool, optional): _description_. Defaults to False.

        Returns:
            fig, ax: matplotlib fig and ax
        """
        
        if fig is None:
            fig, ax = plt.subplots()

        # generate k-path and labels
        (k_vec, k_dist, k_node) = self.model.k_path(k_path, nk, report=False)
        # diagonalize model on path
        evals, evecs = self.model.solve_ham(k_vec, return_eigvecs=True)
        # evecs = np.transpose(evecs, axes=(1, 0, 2)) # [k, n, orb]
        # evals = np.transpose(evals, axes=(1, 0)) # [k, n]
        n_eigs = evecs.shape[1]

        # scattered bands with sublattice color
        if red_lat_idx is not None:
            wt = abs(evecs)**2
            col = np.sum([  wt[..., i] for i in red_lat_idx ], axis=0)

            for n in range(n_eigs):
                scat = ax.scatter(k_dist, evals[:, n], c=col[:, n], cmap='bwr', marker='o', s=scat_size, vmin=0, vmax=1, zorder=2)

            cbar = fig.colorbar(scat, ticks=[1,0])
            cbar.ax.set_yticklabels([r'$\psi_1$', r'$\psi_2$'], size=12)

        else:
            # continuous bands
            for n in range(n_eigs):
                ax.plot(k_dist, evals[:, n], c=lc, lw=lw)

        ax.set_xlim(0, k_node[-1])
        ax.set_xticks(k_node)
        for n in range(len(k_node)):
            ax.axvline(x=k_node[n], linewidth=0.5, color='k', zorder=1)
        if k_label is not None:
            ax.set_xticklabels(k_label, size=12)
        
        ax.set_title(title)
        ax.set_ylabel(r"Energy $E(\mathbf{{k}})$", size=12)
        ax.yaxis.labelpad = 10

        if save_name is not None:
            plt.savefig(save_name)

        if show:
            plt.show()

        return fig, ax
        

class Wannier():
    def __init__(
            self, model: Model, nks: list  
            ):
        self.model: Model = model
        self._nks: list = nks
        self.K_mesh: K_mesh = K_mesh(model, *nks)

        self.energy_eigstates: Bloch = Bloch(model, *nks)
        self.energy_eigstates.solve_model()
        self.tilde_states: Bloch = Bloch(model, *nks)

        self.supercell = list(product(*[range(-int((nk-nk%2)/2), int((nk-nk%2)/2)) for nk in nks]))  # used for real space looping of WFs

    def get_tilde_states(self):
        return self.tilde_states.get_states()
    
    def get_Bloch_Ham(self):
        return self.tilde_states.get_Bloch_Ham() 

    def get_centers(self, Cartesian=False):
        if Cartesian:
            return self.centers
        else:
            return self.centers @ np.linalg.inv(self.model._lat_vecs)
           
    def get_trial_wfs(self, tf_list):
        """
        Args:
            tf_list: list[int | list[tuple]] | list["random", int]
                list of numbers or tuples defining either the integer site
                of the trial function (delta) or the tuples (site, amplitude)
    
        Returns:
            tfs (num_tf x norb np.ndarray): 2 dimensional array of trial functions
        """
        if tf_list[0] == "random":
            n_tfs = tf_list[1]
            def gram_schmidt(vectors):
                orthogonal_vectors = []
                for v in vectors:
                    for u in orthogonal_vectors:
                        v -= np.dot(v, u) * u
                    norm = np.linalg.norm(v)
                    if norm > 1e-10:
                        orthogonal_vectors.append(v / norm)
                return np.array(orthogonal_vectors)

            # Generate n_tfs random n_orb-dimensional vectors
            vectors = abs(np.random.randn(n_tfs, self.model._n_orb))
            # Apply Gram-Schmidt to orthonormalize them
            orthonorm_vecs = gram_schmidt(vectors)
            return orthonorm_vecs

        # number of trial functions to define
        num_tf = len(tf_list)

        # initialize array containing tfs = "trial functions"
        tfs = np.zeros([num_tf, self.model._n_orb], dtype=complex)
        for j, tf in enumerate(tf_list):
            if isinstance(tf, (int, np.int64)):
                # trial function only has weight on one site
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
    
    
    def set_tilde_states(self, tilde_states, cell_periodic=False):
        # set tilde states
        self.tilde_states.set_wfs(tilde_states, cell_periodic=cell_periodic)

        # Fourier transform Bloch-like states to set WFs
        psi_wfs = self.tilde_states._psi_wfs
        dim_k = len(psi_wfs.shape[:-2])
        self.WFs = np.fft.ifftn(psi_wfs, axes=[i for i in range(dim_k)], norm=None)

        # set spreads
        spread = self.spread_recip(decomp=True)
        self.spread = spread[0][0]
        self.omega_i = spread[0][1]
        self.omega_til = spread[0][2]
        self.centers = spread[1]

    
    def tf_overlap_mat(self, psi_wfs, tfs, state_idx):
        """
        Returns A_{k, n, j} = <psi_{n,k} | t_{j}> where psi are Bloch states and t are
        the trial wavefunctions.

        Args:
            psi_wfs (np.array): Bloch eigenstates
            tfs (np.array): trial wfs
            state_idx (list): band indices to form overlap matrix with

        Returns:
            A (np.array): overlap matrix
        """
        psi_wfs = np.take(psi_wfs, state_idx, axis=-2)
        A = np.einsum("...ij, kj -> ...ik", psi_wfs.conj(), tfs)
        return A
    
    
    def get_psi_tilde(self, psi_wfs, tfs, state_idx, compact_SVD=False):
        A = self.tf_overlap_mat(psi_wfs, tfs, state_idx=state_idx)
        V, S, Wh = np.linalg.svd(A, full_matrices=False)

        # TODO: Test this method
        if compact_SVD: 
            V = V[..., :, :-1]
            S = S[..., :-1]
            Wh = Wh[..., :-1, :]

        # swap only last two indices in transpose (ignore k indices)
        # slice psi_wf to keep only occupied bands
        psi_tilde = (V @ Wh).transpose(
            *([i for i in range(self.K_mesh.dim)] + [self.K_mesh.dim + 1, self.K_mesh.dim])
        ) @ psi_wfs[..., state_idx, :]  # [*nk, nband, norb]

        return psi_tilde
    
    
    def single_shot(self, tf_list: list, band_idxs: list | None = None, tilde=False):
        """
        Sets the Wannier functions in home unit cell with associated spreads, centers, trial functions 
        and Bloch-like states using the single shot projection method.

        Args:
            tf_list (list): List of tuples with sites and weights. Can be un-normalized. 
            band_idxs (list | None): Band indices to Wannierize. Defaults to occupied bands (lower half).
        Returns:
            w_0n (np.array): Wannier functions in home unit cell
        """

        self.tf_list = tf_list
        self.trial_wfs = self.get_trial_wfs(tf_list)

        if tilde:
            if band_idxs is None:  # assume we are projecting onto all tilde states
                band_idxs = list(range(self.tilde_states._psi_wfs.shape[-2]))

            psi_til_til = self.get_psi_tilde(
                self.tilde_states._psi_wfs, self.trial_wfs, state_idx=band_idxs
                )
            self.set_tilde_states(psi_til_til, cell_periodic=False)

        if not tilde:
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

        r_n = np.zeros((n_wfs, 2), dtype=complex)  # <\vec{r}>_n
        rsq_n = np.zeros(n_wfs, dtype=complex)  # <r^2>_n
        R_nm = np.zeros((2, n_wfs, n_wfs, nx * ny), dtype=complex)

        expc_rsq = 0  # <r^2>
        expc_r_sq = 0  # <\vec{r}>^2

        for n in range(n_wfs):  # Wannier index
            for tx, ty in self.supercell:  # cells in supercell
                for i, orb in enumerate(self.model._orb_vecs):  # values of Wannier function on lattice
                    pos = (orb[0] + tx) * self.model._lat_vecs[0] + (orb[1] + ty) * self.model._lat_vecs[1]  # position
                    r = np.sqrt(pos[0] ** 2 + pos[1] ** 2)

                    w0n_r = w0[tx, ty, n, i]  # Wannier function

                    # expectation value of position (vector)
                    r_n[n, :] += abs(w0n_r) ** 2 * pos
                    rsq_n[n] += r**2 * w0n_r * w0n_r.conj()

                    if decomp:
                        for m in range(n_wfs):
                            for j, [dx, dy] in enumerate(self.supercell):
                                #TODO: shouldn't this be indexed by j? 
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
                    np.diagonal(R_nm[:, :, :, self.supercell.index((0, 0))], axis1=1, axis2=2)
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
    
    
    def get_Omega_I(self, tilde=True):
        if tilde:
            P = self.tilde_states.get_projector()
            _, Q_nbr = self.tilde_states.get_nbr_projector(return_Q=True)
        else:
            P = self.energy_eigstates.get_projector()
            _, Q_nbr = self.energy_eigstates.get_nbr_projector(return_Q=True)

        nks = self.K_mesh.nks
        Nk = np.prod(nks)
        num_nnbrs = self.K_mesh.num_nnbrs
        w_b, _, idx_shell = self.K_mesh.get_weights(N_sh=1)

        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
            T_kb[..., idx] = np.trace(P[..., :, :] @ Q_nbr[..., idx, :, :], axis1=-1, axis2=-2)

        return (1 / Nk) * w_b[0] * np.sum(T_kb)
    
    def get_Omega_I_k(self, tilde=True):
        if tilde:
            P = self.tilde_states.get_projector()
            _, Q_nbr = self.tilde_states.get_nbr_projector(return_Q=True)
        else:
            P = self.energy_eigstates.get_projector()
            _, Q_nbr = self.energy_eigstates.get_nbr_projector(return_Q=True)
    
        nks = self.K_mesh.nks
        Nk = np.prod(nks)
        num_nnbrs = self.K_mesh.num_nnbrs
        w_b, _, idx_shell = self.K_mesh.get_weights(N_sh=1)

        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
            T_kb[..., idx] = np.trace(P[..., :, :] @ Q_nbr[..., idx, :, :], axis1=-1, axis2=-2)

        return (1 / Nk) * w_b[0] * np.sum(T_kb, axis=-1)
    
    def get_omega_I_k(self):
        P = self.tilde_states.get_projector()
        P_nbr, Q_nbr = self.tilde_states.get_nbr_projector(return_Q=True)
        nks = self.K_mesh.nks
        Nk = np.prod(nks)
        num_nnbrs = self.K_mesh.num_nnbrs
        w_b, _, idx_shell = self.K_mesh.get_weights(N_sh=1)

        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
            T_kb[..., idx] = np.trace(P[..., :, :] @ Q_nbr[..., idx, :, :], axis1=-1, axis2=-2)

        return (1 / Nk) * w_b[0] * np.sum(T_kb, axis=-1)
    
     ####### Maximally Localized WF #######

    def find_optimal_subspace(
        self, N_wfs=None, inner_window=None, outer_window="occupied", 
        iter_num=100, verbose=False, tol=1e-10, alpha=1
    ):
        # useful constants
        nks = self._nks 
        Nk = np.prod(nks)
        n_orb = self.model._n_orb
        n_occ = int(n_orb/2)

        # eigenenergies and eigenstates for inner/outer window
        energies = self.energy_eigstates.get_energies()
        unk_states = self.energy_eigstates.get_states()["Cell periodic"]

        # initial subspace
        init_states = self.tilde_states
      
        #### Setting inner/outer energy windows ####

        # number of states in target manifold 
        if N_wfs is None:
            N_wfs = init_states._n_states

        # outer window
        if outer_window == "occupied":
            outer_window_type = "bands" # optimally would like to use band indices

            # used in case inner window is defined by energy values
            outer_band_idxs = list(range(n_occ))
            outer_band_energies = energies[..., outer_band_idxs]
            outer_energies = [np.argmin(outer_band_energies), np.argmax(outer_band_energies)]

            # mask out states outside outer window
            nan = np.empty(unk_states.shape)
            nan.fill(np.nan)
            states_sliced = np.where(
                np.logical_and(
                    energies[..., np.newaxis] >= outer_energies[0], 
                    energies[..., np.newaxis] <= outer_energies[1]
                    ), 
                    unk_states, nan)
            mask_outer = np.isnan(states_sliced)
            masked_outer_states = np.ma.masked_array(states_sliced, mask=mask_outer)

        elif list(outer_window.keys())[0].lower() == 'bands':
            outer_window_type = "bands"

            # used in case inner window is defined by energy values
            outer_band_idxs = list(outer_window.values())[0]
            outer_band_energies = energies[..., outer_band_idxs]
            outer_energies = [np.argmin(outer_band_energies), np.argmax(outer_band_energies)]

            # mask out states outside outer window
            nan = np.empty(unk_states.shape)
            nan.fill(np.nan)
            states_sliced = np.where(
                np.logical_and(
                    energies[..., np.newaxis] >= outer_energies[0], 
                    energies[..., np.newaxis] <= outer_energies[1]
                    ), 
                    unk_states, nan)
            mask_outer = np.isnan(states_sliced)
            masked_outer_states = np.ma.masked_array(states_sliced, mask=mask_outer)

        elif list(outer_window.keys())[0].lower() == 'energy':
            outer_window_type = "energy"

            # energy window
            outer_energies = np.sort(list(outer_window.values())[0])

            # mask out states outside outer window
            nan = np.empty(unk_states.shape)
            nan.fill(np.nan)
            states_sliced = np.where(
                np.logical_and(
                    energies[..., np.newaxis] >= outer_energies[0], 
                    energies[..., np.newaxis] <= outer_energies[1]
                    ), 
                    unk_states, nan)
            mask_outer = np.isnan(states_sliced)
            masked_outer_states = np.ma.masked_array(states_sliced, mask=mask_outer)
            
        # inner window
        if inner_window is None:
            N_inner = 0
            inner_window_type = outer_window_type
            inner_band_idxs = None

        elif list(inner_window.keys())[0].lower() == 'bands':
            inner_window_type = "bands"

            inner_band_idxs = list(inner_window.values())[0]
            inner_band_energies = energies[..., inner_band_idxs]
            inner_energies = [np.argmin(inner_band_energies), np.argmax(inner_band_energies)]

            # used in case outer window is energy dependent
            nan = np.empty(unk_states.shape)
            nan.fill(np.nan)
            states_sliced = np.where(
                np.logical_and(
                    energies[..., np.newaxis] >= inner_energies[0], 
                    energies[..., np.newaxis] <= inner_energies[1]), 
                    unk_states, nan
                    )
            mask_inner = np.isnan(states_sliced)
            masked_inner_states = np.ma.masked_array(states_sliced, mask=mask_inner)
            inner_states = masked_inner_states

        elif list(inner_window.keys())[0].lower() == 'energy':
            inner_window_type = "energy"

            inner_energies =  np.sort(list(inner_window.values())[0])

            nan = np.empty(unk_states.shape)
            nan.fill(np.nan)
            states_sliced = np.where(
                np.logical_and(
                    energies[..., np.newaxis] >= inner_energies[0], 
                    energies[..., np.newaxis] <= inner_energies[1]), 
                    unk_states, nan
                    )
            mask_inner = np.isnan(states_sliced)
            masked_inner_states = np.ma.masked_array(states_sliced, mask=mask_inner)
            inner_states = masked_inner_states
            N_inner = (~inner_states.mask).sum(axis=(-1,-2))//n_orb

        if inner_window_type == "bands" and outer_window_type == "bands":
            # defer to the faster function
            return self.find_optimal_subspace_bands(
                N_wfs=N_wfs, inner_bands=inner_band_idxs, outer_bands=outer_band_idxs, 
                iter_num=iter_num, verbose=verbose, tol=tol, alpha=alpha)

        # minimization manifold
        if inner_window is not None:
            # states in outer manifold and outside inner manifold
            min_mask = ~np.logical_and(~mask_outer, mask_inner)
            min_states = np.ma.masked_array(unk_states, mask=min_mask)
            min_states = np.ma.filled(min_states, fill_value=0)
        else:
            min_states = masked_outer_states
            min_states = np.ma.filled(min_states, fill_value=0)
        
        # number of states to keep in minimization subspace
        if inner_window is None:
            # keep all the states from minimization manifold
            num_keep = np.full(min_states.shape, N_wfs)
            keep_mask = (np.arange(min_states.shape[-2]) >= num_keep)
            keep_mask = np.swapaxes(keep_mask, axis1=-1, axis2=-2)
        else: # n_inner is k-dependent when using energy window
            N_inner = (~inner_states.mask).sum(axis=(-1,-2))//n_orb
            num_keep = N_wfs - N_inner # matrix of integers
            keep_mask = (np.arange(min_states.shape[-2]) >= (num_keep[:, :, np.newaxis, np.newaxis]))
            keep_mask = keep_mask.repeat(min_states.shape[-2], axis=-2)
            keep_mask = np.swapaxes(keep_mask, axis1=-1, axis2=-2)

        N_min = (~min_states.mask).sum(axis=(-1,-2))//n_orb
        N_outer = (~masked_outer_states.mask).sum(axis=(-1,-2))//n_orb

        # Assumes only one shell for now
        w_b, _, idx_shell = self.K_mesh.get_weights(N_sh=1)
        num_nnbrs = self.K_mesh.num_nnbrs
        bc_phase = self.K_mesh.bc_phase
        
        # Projector of initial tilde subspace at each k-point
        P = init_states.get_projector()
        P_nbr, Q_nbr = init_states.get_nbr_projector(return_Q=True)
        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
            T_kb[..., idx] = np.trace(P[..., :, :] @ Q_nbr[..., idx, :, :], axis1=-1, axis2=-2)
        P_min = np.copy(P)  # for start of iteration
        P_nbr_min = np.copy(P_nbr)  # for start of iteration
        Q_nbr_min = np.copy(Q_nbr)  # for start of iteration

        omega_I_prev = (1 / Nk) * w_b[0] * np.sum(T_kb)

        #### Start of minimization iteration ####
        for i in range(iter_num):
            P_avg = np.sum(w_b[0] * P_nbr_min, axis=-3)
            Z = min_states.conj() @ P_avg @ np.transpose(min_states, axes=(0,1,3,2))
            # masked entries correspond to subspace spanned by states outside min manifold
            Z = np.ma.filled(Z, fill_value=0)

            eigvals, eigvecs = np.linalg.eigh(Z) # [..., val, idx]
            eigvecs = np.swapaxes(eigvecs, axis1=-1, axis2=-2) # [..., idx, val]

            # eigvals = 0 correspond to states outside the minimization manifold. Mask these out.
            zero_mask = eigvals.round(10) == 0
            non_zero_eigvals = np.ma.masked_array(eigvals, mask=zero_mask)
            non_zero_eigvecs = np.ma.masked_array(eigvecs, mask=np.repeat(zero_mask[..., np.newaxis], repeats=eigvals.shape[-1], axis=-1))

            # sort eigvals and eigvecs by eigenvalues in descending order excluding eigvals=0
            sorted_eigvals_idxs = np.argsort(-non_zero_eigvals, axis=-1)
            sorted_eigvals = np.take_along_axis(non_zero_eigvals, sorted_eigvals_idxs, axis=-1)
            sorted_eigvecs = np.take_along_axis(non_zero_eigvecs, sorted_eigvals_idxs[..., np.newaxis], axis=-2)
            sorted_eigvecs = np.ma.filled(sorted_eigvecs, fill_value=0)

            states_min = np.einsum('...ji, ...ik->...jk', sorted_eigvecs, min_states)
            keep_states = np.ma.masked_array(states_min, mask=keep_mask)
            keep_states = np.ma.filled(keep_states, fill_value=0)
            # need to concatenate with frozen states

            P_new = np.einsum("...ni,...nj->...ij", keep_states, keep_states.conj())
            P_min = alpha * P_new + (1 - alpha) * P_min # for next iteration
            for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
                states_pbc = np.roll(keep_states, shift=tuple(-idx_vec), axis=(0,1)) * bc_phase[..., idx, np.newaxis,  :]
                P_nbr_min[..., idx, :, :] = np.einsum(
                        "...ni, ...nj->...ij", states_pbc, states_pbc.conj()
                        )
                Q_nbr_min[..., idx, :, :] = np.eye(n_orb) - P_nbr_min[..., idx, :, :]
                T_kb[..., idx] = np.trace(P_min[..., :, :] @ Q_nbr_min[..., idx, :, :], axis1=-1, axis2=-2)
            
            omega_I_new = (1 / Nk) * w_b[0] * np.sum(T_kb)
            diff = omega_I_prev - omega_I_new
            omega_I_prev = omega_I_new

            if verbose and diff > 0:
                print("Warning: Omega_I is increasing.")

            if verbose:
                print(f"{i} Omega_I: {omega_I_new.real}")
            
            if abs(diff) * (iter_num - i) <= tol:
                # assuming the change in omega_i monatonically decreases at this rate, 
                # omega_i will not change more than tolerance with remaining steps
                print("Omega_I has converged within tolerance. Breaking loop")
                if inner_window is not None:
                    min_keep = np.ma.masked_array(keep_states, mask=keep_mask)
                    subspace = np.ma.concatenate((min_keep, inner_states), axis=-2)
                    subspace_sliced = subspace[np.where(~subspace.mask)]
                    subspace_sliced = subspace_sliced.reshape((*nks, N_wfs, n_orb))
                    subspace_sliced = np.array(subspace_sliced)
                    return subspace_sliced
                else:
                    return keep_states

        # loop has ended
        if inner_window is not None:
            min_keep = np.ma.masked_array(keep_states, mask=keep_mask)
            subspace = np.ma.concatenate((min_keep, inner_states), axis=-2)
            subspace_sliced = subspace[np.where(~subspace.mask)]
            subspace_sliced = subspace_sliced.reshape((*nks, N_wfs, n_orb))
            subspace_sliced = np.array(subspace_sliced)
            return subspace_sliced
        else:
            return keep_states  


    def find_optimal_subspace_bands(
        self, N_wfs=None, inner_bands=None, outer_bands="occupied", 
        iter_num=100, verbose=False, tol=1e-10, alpha=1
    ):
        """Finds the subspaces throughout the BZ that minimizes the gauge-independent spread. 

        Used when the inner and outer windows correspond to bands rather than energy values. This function
        is faster when compared to energy windows. By specifying bands, the arrays have fixed sizes at each k-point
        and operations can be vectorized with numpy. 
        """
        nks = self._nks 
        Nk = np.prod(nks)
        n_orb = self.model._n_orb
        n_occ = int(n_orb/2)

        # Assumes only one shell for now
        w_b, _, idx_shell = self.K_mesh.get_weights(N_sh=1)
        bc_phase = self.K_mesh.bc_phase

        # initial subspace
        init_states = self.tilde_states

        if N_wfs is None:
            # assume number of states in the subspace is number of tilde states 
            N_wfs = init_states._n_states

        if outer_bands == "occupied":
            outer_bands = list(range(n_occ))

        outer_states = self.energy_eigstates._u_wfs.take(outer_bands, axis=-2)

        # Projector of initial tilde subspace at each k-point
        if inner_bands is None:
            N_inner = 0
            P = init_states.get_projector()
            P_nbr, Q_nbr = init_states.get_nbr_projector(return_Q=True)
            T_kb = np.einsum('...ij, ...kji -> ...k', P, Q_nbr)
        else:
            N_inner = len(inner_bands)
            inner_states = self.energy_eigstates._u_wfs.take(inner_bands, axis=-2)

            P_inner = np.swapaxes(inner_states, -1, -2) @ inner_states.conj()
            Q_inner = np.eye(P_inner.shape[-1]) - P_inner
            P_tilde = self.tilde_states.get_projector()

            # chosing initial subspace as highest eigenvalues 
            MinMat = Q_inner @ P_tilde @ Q_inner
            eigvals, eigvecs = np.linalg.eigh(MinMat)
            min_states = np.einsum('...ij, ...ik->...jk', eigvecs[..., -(N_wfs-N_inner):], outer_states)

            P = np.swapaxes(min_states,-1, -2) @ min_states.conj()
            states_pbc_all = np.empty((*min_states.shape[:-2], len(idx_shell[0]), *min_states.shape[-2:]), dtype=min_states.dtype)
            for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
                states_pbc_all[..., idx, :, :] = np.roll(min_states, shift=tuple(-idx_vec), axis=(0,1)) * bc_phase[..., idx, np.newaxis,  :]
            P_nbr = np.swapaxes(states_pbc_all, -1, -2) @ states_pbc_all.conj()
            Q_nbr = np.eye(n_orb) - P_nbr
            T_kb = np.einsum('...ij, ...kji -> ...k', P, Q_nbr)

        P_min = np.copy(P)  # for start of iteration
        P_nbr_min = np.copy(P_nbr)  # for start of iteration

        # manifold from which we borrow states to minimize omega_i
        comp_bands = list(np.setdiff1d(outer_bands, inner_bands))
        comp_states = self.energy_eigstates._u_wfs[..., comp_bands, :]

        omega_I_prev = (1 / Nk) * w_b[0] * np.sum(T_kb)

        for i in range(iter_num):
            # states spanning optimal subspace minimizing gauge invariant spread
            P_avg = w_b[0] * np.sum(P_nbr_min, axis=-3)
            Z = comp_states.conj() @ P_avg @ np.swapaxes(comp_states, -1, -2)
            eigvals, eigvecs = np.linalg.eigh(Z) # [val, idx]
            evecs_keep = eigvecs[..., -(N_wfs-N_inner):]
            states_min = np.swapaxes(evecs_keep, -1, -2) @ comp_states

            P_new = np.swapaxes(states_min,-1, -2) @ states_min.conj()

            states_pbc_all = np.empty((*states_min.shape[:-2], len(idx_shell[0]), *states_min.shape[-2:]), dtype=states_min.dtype)
            for idx, idx_vec in enumerate(idx_shell[0]):  # nearest neighbors
                states_pbc_all[..., idx, :, :] = np.roll(states_min, shift=tuple(-idx_vec), axis=(0,1)) * bc_phase[..., idx, np.newaxis,  :]
            P_nbr_new = np.swapaxes(states_pbc_all, -1, -2) @ states_pbc_all.conj()

            if alpha != 1:
                # for next iteration
                P_min = alpha * P_new + (1 - alpha) * P_min 
                P_nbr_min = alpha * P_nbr_new + (1 - alpha) * P_nbr_min 
            else:
                # for next iteration
                P_min = P_new  
                P_nbr_min = P_nbr_new 

            if verbose:
                Q_nbr_min = np.eye(n_orb) - P_nbr_min 
                T_kb = np.einsum('...ij, ...kji -> ...k', P_min, Q_nbr_min)
                omega_I_new = (1 / Nk) * w_b[0] * np.sum(T_kb)

                if omega_I_new > omega_I_prev:
                    print("Warning: Omega_I is increasing.")
                    alpha = max(alpha - 0.1, 0)

                if abs(omega_I_prev - omega_I_new) * (iter_num - i) <= tol:
                    # omega_i will not change by more than tol with remaining steps (if monotonically decreases)
                    print("Omega_I has converged within tolerance. Breaking loop")
                    break

                print(f"{i} Omega_I: {omega_I_new.real}")
                omega_I_prev = omega_I_new

            else:
                if abs(np.amax(P_new - P_min)) <= tol:
                    print(np.amax(P_new - P_min))
                    break

        if inner_bands is not None:
            return_states = np.concatenate((inner_states, states_min), axis=-2)
            return return_states
        else:
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
    
    
    def find_min_unitary(
            self, eps=1e-3, iter_num=100, verbose=False, tol=1e-10, grad_min=1e-3
        ):
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
        """
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
            r_n = -(1 / Nk) * w_b * np.sum(
                log_diag_M_imag:=np.log(np.diagonal(M, axis1=-1, axis2=-2)).imag, axis=(0,1)).T @ k_shell
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

            if verbose:
                print(
                    f"{i} Omega_til = {omega_tilde_new.real}, Grad mag: {grad_mag}"
                )

            if abs(grad_mag) <= grad_min and abs(omega_tilde_prev - omega_tilde_new) * (iter_num - i) <= tol:
                print("Omega_tilde minimization has converged within tolerance. Breaking the loop.")
                return U
            
            if grad_mag_prev < grad_mag:
                if verbose:
                    print("Warning: Gradient increasing.")
                eps *= 0.9
            
            grad_mag_prev = grad_mag
            omega_tilde_prev = omega_tilde_new

        return U
    

    def subspace_selec(
        self, outer_window="occupied", inner_window=None, twfs=None,
        N_wfs=None, iter_num=1000, tol=1e-5, alpha=1, verbose=False
    ):
        # if we haven't done single-shot projection yet (set tilde states)
        if twfs is not None:
            # initialize tilde states
            twfs = self.get_trial_wfs(twfs)

            n_occ = int(self.energy_eigstates._n_states / 2)  # assuming half filled
            band_idxs = list(range(0, n_occ)) # project onto occ manifold

            psi_til_init = self.get_psi_tilde(
                self.energy_eigstates._psi_wfs, twfs, state_idx=band_idxs)
            self.set_tilde_states(psi_til_init, cell_periodic=False)
        else:
            assert hasattr(self.tilde_states, "_u_wfs"), "Need pass trial wavefunction list or initalize tilde states with single_shot()."
        
        # Minimizing Omega_I via disentanglement
        util_min = self.find_optimal_subspace(
            N_wfs=N_wfs,
            outer_window=outer_window,
            inner_window=inner_window,
            iter_num=iter_num,
            verbose=verbose, alpha=alpha, tol=tol
        )

        self.set_tilde_states(util_min, cell_periodic=True)

        return
    
    def max_loc(
        self, eps=1e-3, iter_num=1000, tol=1e-5, grad_min=1e-3, verbose=False   
    ):

        U = self.find_min_unitary(
            eps=eps, iter_num=iter_num, verbose=verbose, tol=tol, grad_min=grad_min)
        
        u_tilde_wfs = self.tilde_states._u_wfs
        util_max_loc = np.einsum('...ji, ...jm -> ...im', U, u_tilde_wfs)
        
        self.set_tilde_states(util_max_loc, cell_periodic=True)
        
        return

    def ss_maxloc(
        self,
        outer_window="occupied",
        inner_window=None,
        twfs_1=None,
        twfs_2=None,
        N_wfs=None,
        iter_num_omega_i=1000,
        iter_num_omega_til=1000,
        eps=1e-3,
        tol_omega_i=1e-5,
        tol_omega_til=1e-10,
        grad_min=1e-3,
        alpha=1,
        verbose=False,
    ):
        """ Find the maximally localized Wannier functions using the projection method.
        """

        ### Subspace selection ###
        self.subspace_selec(
            outer_window=outer_window,
            inner_window=inner_window, 
            twfs=twfs_1, 
            N_wfs=N_wfs, 
            iter_num=iter_num_omega_i,
            tol=tol_omega_i, 
            alpha=alpha, 
            verbose=verbose
        )

        ### Second projection ###
        # if we need a smaller number of twfs b.c. of subspace selec
        if twfs_2 is not None:
            twfs = self.get_trial_wfs(twfs_2)
            psi_til_til = self.get_psi_tilde(
                self.tilde_states._psi_wfs, twfs, 
                state_idx=list(range(self.tilde_states._psi_wfs.shape[2]))
            )
        # chose same twfs as in subspace selec
        else:
            psi_til_til = self.get_psi_tilde(
                    self.tilde_states._psi_wfs, self.trial_wfs, 
                    state_idx=list(range(self.tilde_states._psi_wfs.shape[2]))
                )

        self.set_tilde_states(psi_til_til, cell_periodic=False)
    
        ### Finding optimal gauge with maxloc ###
        self.max_loc(
            eps=eps, 
            iter_num=iter_num_omega_til, 
            tol=tol_omega_til,
            grad_min=grad_min,
            verbose=verbose
            )
        
        return
    

    def interp_energies(self, k_path, wan_idxs=None, ret_eigvecs=False, u_tilde=None):
        if u_tilde is None:
            u_tilde = self.get_tilde_states()["Cell periodic"]
        if wan_idxs is not None:
            u_tilde = np.take_along_axis(u_tilde, wan_idxs, axis=-2)

        H_k = self.get_Bloch_Ham()
        H_rot_k = u_tilde[..., :, :].conj() @ H_k @ np.transpose(u_tilde[..., : ,:], axes=(0,1,3,2))

        k_mesh = self.K_mesh.full_mesh
        k_idx_arr = self.K_mesh.idx_arr
        Nk = np.prod([self.K_mesh.nks])

        nks = self.K_mesh.nks
        supercell = list(product(*[range(-int((nk-nk%2)/2), int((nk-nk%2)/2)+1) for nk in nks]))
        # supercell = list(product(*[range(-int((nk-nk%2)/2), int((nk-nk%2)/2)+1) for nk in nks]))

        # Fourier transform to real space
        H_R = np.zeros((len(supercell), H_rot_k.shape[-1], H_rot_k.shape[-1]), dtype=complex)
        u_R = np.zeros((len(supercell), u_tilde.shape[-2], u_tilde.shape[-1]), dtype=complex)
        for idx, (x, y) in enumerate(supercell):
            for k_idx in k_idx_arr:
                R_vec = np.array([x, y])
                phase = np.exp(-1j * 2 * np.pi * np.vdot(k_mesh[k_idx], R_vec))
                H_R[idx, :, :] += H_rot_k[k_idx] * phase / Nk
                u_R[idx] += u_tilde[k_idx] * phase / Nk

        # interpolate to arbitrary k
        H_k_interp = np.zeros((k_path.shape[0], H_R.shape[-1], H_R.shape[-1]), dtype=complex)
        u_k_interp = np.zeros((k_path.shape[0], u_R.shape[-2], u_R.shape[-1]), dtype=complex)
        for k_idx, k in enumerate(k_path):
            for idx, (x, y) in enumerate(supercell):
                R_vec = np.array([x, y])
                phase = np.exp(1j * 2 * np.pi * np.vdot(k, R_vec))
                H_k_interp[k_idx] += H_R[idx] * phase
                u_k_interp[k_idx] += u_R[idx] * phase

        eigvals, eigvecs = np.linalg.eigh(H_k_interp)
        eigvecs = np.einsum('...ij, ...ik->...jk', u_k_interp, eigvecs)
        eigvecs = np.transpose(eigvecs, axes=(0, 2, 1))

        if ret_eigvecs:
            return eigvals, eigvecs
        else:
            return eigvals
    
    
    def interp_op(self, O_k, k_path, plaq=False):
        return self.energy_eigstates.interp_op(O_k, k_path, plaq=plaq)
       

    def report(self):
        assert hasattr(self.tilde_states, '_u_wfs'), "First need to set Wannier functions"
        print("Wannier function report")
        print(" --------------------- ")

        print("Quadratic spreads:")
        for i, spread in enumerate(self.spread):
            print(f"w_{i} --> {spread.round(5)}")
        print("Centers:")
        centers = self.get_centers()
        for i, center in enumerate(centers):
            print(f"w_{i} --> {center.round(5)}")
        print(rf"Omega_i = {self.omega_i}")
        print(rf"Omega_tilde = {self.omega_til}")

        
    def get_supercell(self, Wan_idx, omit_sites=None):
        w0 = self.WFs
        center = self.centers[Wan_idx]
        orbs = self.model._orb_vecs
        lat_vecs = self.model._lat_vecs
        
        # Initialize arrays to store positions and weights
        positions = {
            'all': {'xs': [], 'ys': [], 'r': [], 'wt': [], 'phase': []},
            'home even': {'xs': [], 'ys': [], 'r': [], 'wt': [], 'phase': []},
            'home odd': {'xs': [], 'ys': [], 'r': [], 'wt': [], 'phase': []},
            'omit': {'xs': [], 'ys': [], 'r': [], 'wt': [], 'phase': []},
            'even': {'xs': [], 'ys': [], 'r': [], 'wt': [], 'phase': []},
            'odd': {'xs': [], 'ys': [], 'r': [], 'wt': [], 'phase': [] }
        }

        for tx, ty in self.supercell:
            for i, orb in enumerate(orbs):
                # Extract relevant parameters
                wf_value = w0[tx, ty, Wan_idx, i]
                phase = np.arctan2(wf_value.imag, wf_value.real) 
                wt = np.abs(wf_value) ** 2
                pos = orb[0] * lat_vecs[0] + tx * lat_vecs[0] + orb[1] * lat_vecs[1] + ty * lat_vecs[1]
                rel_pos = pos - center
                x, y, rad = pos[0], pos[1], np.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2)

                # Store values in 'all'
                positions['all']['xs'].append(x)
                positions['all']['ys'].append(y)
                positions['all']['r'].append(rad)
                positions['all']['wt'].append(wt)
                positions['all']['phase'].append(phase)

                # Handle omit site if applicable
                if omit_sites is not None and i in omit_sites:
                    positions['omit']['xs'].append(x)
                    positions['omit']['ys'].append(y)
                    positions['omit']['r'].append(rad)
                    positions['omit']['wt'].append(wt)
                    positions['omit']['phase'].append(phase)
                # Separate even and odd index sites
                if i % 2 == 0:
                    positions['even']['xs'].append(x)
                    positions['even']['ys'].append(y)
                    positions['even']['r'].append(rad)
                    positions['even']['wt'].append(wt)
                    positions['even']['phase'].append(phase)
                    if tx == ty == 0:
                        positions['home even']['xs'].append(x)
                        positions['home even']['ys'].append(y)
                        positions['home even']['r'].append(rad)
                        positions['home even']['wt'].append(wt)
                        positions['home even']['phase'].append(phase)

                else:
                    positions['odd']['xs'].append(x)
                    positions['odd']['ys'].append(y)
                    positions['odd']['r'].append(rad)
                    positions['odd']['wt'].append(wt)
                    positions['odd']['phase'].append(phase)
                    if tx == ty == 0:
                        positions['home odd']['xs'].append(x)
                        positions['home odd']['ys'].append(y)
                        positions['home odd']['r'].append(rad)
                        positions['home odd']['wt'].append(wt)
                        positions['home odd']['phase'].append(phase)


        # Convert lists to numpy arrays (batch processing for cleanliness)
        for key, data in positions.items():
            for sub_key in data:
                positions[key][sub_key] = np.array(data[sub_key])

        self.positions = positions


    def plot_density(
        self, Wan_idx,
        title=None, save_name=None, mark_home_cell=False,
        mark_center=False, show_lattice=False, omit_sites=None,
        show=False, interpolate=False,
        scatter_size=40, lat_size=2, fig=None, ax=None, cbar=True, return_fig=False
        ):

        center = self.centers[Wan_idx]

        if not hasattr(self, "positions"):
            self.get_supercell(Wan_idx, omit_sites=omit_sites)

        positions = self.positions

        # Extract arrays for plotting or further processing
        xs = positions['all']['xs']
        ys = positions['all']['ys']
        w0i_wt = positions['all']['wt']

        xs_ev_home = positions['home even']['xs']
        ys_ev_home = positions['home even']['ys']
        xs_odd_home = positions['home odd']['xs']
        ys_odd_home = positions['home odd']['ys']

        xs_omit = positions['omit']['xs']
        ys_omit = positions['omit']['ys']
        w0omit_wt = positions['omit']['wt']

        xs_ev = positions['even']['xs']
        ys_ev = positions['even']['ys']
        w0ev_wt = positions['even']['wt']

        xs_odd = positions['odd']['xs']
        ys_odd = positions['odd']['ys']
        w0odd_wt = positions['odd']['wt']
          
        if fig is None:
            fig, ax = plt.subplots()

        # Weight plot
        if interpolate:
            from scipy.interpolate import griddata
            grid_x, grid_y = np.mgrid[min(xs):max(xs):2000j, min(ys):max(ys):2000j]
            grid_z = griddata((xs, ys), w0i_wt, (grid_x, grid_y), method='linear')
            dens_plot = plt.pcolormesh(grid_x, grid_y, grid_z, cmap='plasma', norm=LogNorm(vmin=2e-16, vmax=1))
        else:
            dens_plot = ax.scatter(xs, ys, c=w0i_wt, s=scatter_size, cmap='plasma', norm=LogNorm(vmin=2e-16, vmax=1), marker='h', zorder=0)

        if show_lattice:
            scat = ax.scatter(xs_ev, ys_ev, marker='o', c='k', s=lat_size, zorder=2)
            scat = ax.scatter(xs_odd, ys_odd, marker='o', s=lat_size, zorder=2, facecolors='none', edgecolors='k')

        if omit_sites is not None :
            ax.scatter(xs_omit, ys_omit, s=2, marker='x', c='g')

        if mark_home_cell:
            scat = ax.scatter(xs_ev_home, ys_ev_home, marker='o', s=lat_size, zorder=2, facecolors='none', edgecolors='b')
            scat = ax.scatter(xs_odd_home, ys_odd_home, marker='o', s=lat_size, zorder=2, facecolors='none', edgecolors='r')

        if cbar:
            cbar = plt.colorbar(dens_plot, ax=ax)
            # cbar.set_label(rf"$|\langle \phi_{{\vec{{R}}, j}}| w_{{0, {Wan_idx}}}\rangle|^2$", rotation=270)
            cbar.set_label(rf"$|w_{Wan_idx}(\mathbf{{r}} )|^2$", rotation=270)
            cbar.ax.get_yaxis().labelpad = 20

        ax.set_title(title)

        if mark_center:
            ax.scatter(center[0], center[1],
               marker='x', 
               label=fr"Center $\mathbf{{r}}_c = ({center[0]: .3f}, {center[1]: .3f})$", c='g', zorder=1)
            ax.legend(loc='upper right')

        # Saving
        if save_name is not None:
            plt.savefig(f'Wan_wt_{save_name}.png')

        if show:
            plt.show()

        if return_fig:
            return fig, ax
    
    def plot_phase():
        # Phase plot
        #     fig2, ax2 = plt.subplots()
        #     figs.append(fig2)
        #     axs.append(ax2)

        #     scat = ax2.scatter(xs, ys, c=w0i_phase, cmap='hsv')

        #     cbar = plt.colorbar(scat, ax=ax2)
        #     cbar.set_label(
        #         rf"$\phi = \tan^{{-1}}(\mathrm{{Im}}[w_{{0, {Wan_idx}}}(r)]\  / \ \mathrm{{Re}}[w_{{0, {Wan_idx}}}(r)])$", 
        #         rotation=270)
        #     cbar.ax.get_yaxis().labelpad = 20
        #     ax2.set_title(title)

        #     # Saving
        #     if save_name is not None:
        #         plt.savefig(f'Wan_wt_{save_name}.png')
            
        #     if show:
        #         plt.show()
        return
    
    def plot_decay(
            self, Wan_idx, fit_deg=None, fit_rng=None, ylim=None, 
            omit_sites=None, fig=None, ax=None, title=None, show=False, 
            return_fig=True
        ):
        if fig is None:
            fig, ax = plt.subplots()

        if not hasattr(self, "positions"):
            self.get_supercell(Wan_idx, omit_sites=omit_sites)

        # Extract arrays for plotting or further processing
        positions = self.positions
        r = positions['all']['r']
        r_omit = positions['omit']['r']
        r_ev = positions['even']['r']
        r_odd = positions['odd']['r']

        w0i_wt = positions['all']['wt']
        w0omit_wt = positions['omit']['wt']
        w0ev_wt = positions['even']['wt']
        w0odd_wt = positions['odd']['wt']
    
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
        if omit_sites is not None:
            ax.scatter(r_omit[r_omit<cutoff], w0omit_wt[r_omit<cutoff], zorder=1, s=10, c='g', label='omitted site')

        ax.scatter(r[r<cutoff], w0i_wt[r<cutoff], zorder=1, s=10, c='b')

        # ax.scatter(r_ev[r_ev<cutoff], w0ev_wt[r_ev<cutoff], zorder=1, s=10, c='b')
        # ax.scatter(r_odd[r_odd<cutoff], w0odd_wt[r_odd<cutoff], zorder=1, s=10, c='b')

        # bar of avgs
        ax.bar(r_ledge[r_ledge<cutoff], avg_w0i_wt_bins[r_ledge<cutoff], width=1, align='edge', ec='k', zorder=0, ls='-', alpha=0.3)

        # fit line
        if fit_deg is None:
            deg = 1 # polynomial fit degree
        r_fit = r_cntr[np.logical_and(r_cntr > fit_rng[0], r_cntr < fit_rng[1])]
        w0i_wt_fit = avg_w0i_wt_bins[np.logical_and(r_cntr > fit_rng[0], r_cntr < fit_rng[1])]
        fit = np.polyfit(r_fit, np.log(w0i_wt_fit), deg)
        fit_line = np.sum(np.array([r_fit**(deg-i) * fit[i] for i in range(deg+1)]), axis=0)
        fit_label = rf"$Ce^{{{fit[-2]: 0.2f} r  {'+'.join([fr'{c: .2f} r^{deg-j}' for j, c in enumerate(fit[:-3])])}}}$"
        ax.plot(r_fit, np.exp(fit_line), c='lime', ls='--', lw=2.5, label=fit_label)

        ax.legend()
        ax.set_xlabel(r'$|\mathbf{r}- \mathbf{{r}}_c|$', size=12)
        ax.set_ylabel(rf"$|w_{Wan_idx}(\mathbf{{r}}- \mathbf{{r}}_c)|^2$", size=12)
        # ax.set_xlabel(r'$|\vec{R}+\vec{\tau}_j|$')
        # ax.set_xlim(-4e-1, cutoff)
        if ylim is None:
            ax.set_ylim(0.8*min(w0i_wt[r<cutoff]), 1.5)
        else:
            ax.set_ylim(ylim)
        ax.set_yscale('log')

        ax.set_title(title)

        if show:
            plt.show()

        if return_fig:
            return fig, ax
        

    def plot_centers(
        self, omit_sites=None, 
        section_home_cell=True, color_home_cell=True, translate_centers=False,
        title=None, save_name=None, show=False, legend=False, pmx=4, pmy=4,
        kwargs_centers={'s': 80, 'marker': '*', 'c': 'g'},
        kwargs_omit={'s': 50, 'marker': 'x', 'c':'k'},
        kwargs_lat={'s':10, 'marker': 'o', 'c':'k'}, fig=None, ax=None
    ):
        lat_vecs = self.model.get_lat_vecs()
        orbs = self.model.get_orb_vecs(Cartesian=False)
        centers = self.centers

        # Initialize arrays to store positions and weights
        positions = {
            'all': {'xs': [], 'ys': []},
            'centers': {'xs': [[] for i in range(centers.shape[0])], 'ys':[[] for i in range(centers.shape[0])]},
            'home even': {'xs': [], 'ys': []},
            'home odd': {'xs': [], 'ys': []},
            'omit': {'xs': [], 'ys': []},
            'even': {'xs': [], 'ys': []},
            'odd': {'xs': [], 'ys': []}
        }

        for tx, ty in self.supercell:
            for i, orb in enumerate(orbs):
                # Extract relevant parameters
                pos = orb[0] * lat_vecs[0] + tx * lat_vecs[0] + orb[1] * lat_vecs[1] + ty * lat_vecs[1]
                if translate_centers:
                    for j in range(centers.shape[0]):
                        center = centers[j] + tx * lat_vecs[0] + ty * lat_vecs[1]
                        positions['centers']['xs'][j].append(center[0])
                        positions['centers']['ys'][j].append(center[1])

                # rel_pos = pos - center
                x, y = pos[0], pos[1]

                # Store values in 'all'
                positions['all']['xs'].append(x)
                positions['all']['ys'].append(y)

                # Handle omit site if applicable
                if omit_sites is not None and i in omit_sites:
                    positions['omit']['xs'].append(x)
                    positions['omit']['ys'].append(y)
                # Separate even and odd index sites
                if i % 2 == 0:
                    positions['even']['xs'].append(x)
                    positions['even']['ys'].append(y)
                    if tx == ty == 0:
                        positions['home even']['xs'].append(x)
                        positions['home even']['ys'].append(y)
                else:
                    positions['odd']['xs'].append(x)
                    positions['odd']['ys'].append(y)
                    if tx == ty == 0:
                        positions['home odd']['xs'].append(x)
                        positions['home odd']['ys'].append(y)


        # Convert lists to numpy arrays (batch processing for cleanliness)
        for key, data in positions.items():
            for sub_key in data:
                positions[key][sub_key] = np.array(data[sub_key])

        # All positions
        xs = positions['all']['xs']
        ys = positions['all']['ys']

        # home cell site positions
        xs_ev_home = positions['home even']['xs']
        ys_ev_home = positions['home even']['ys']
        xs_odd_home = positions['home odd']['xs']
        ys_odd_home = positions['home odd']['ys']

        # omitted site positions
        xs_omit = positions['omit']['xs']
        ys_omit = positions['omit']['ys']

        # sublattice positions
        xs_ev = positions['even']['xs']
        ys_ev = positions['even']['ys']
        xs_odd = positions['odd']['xs']
        ys_odd = positions['odd']['ys']

        
        if fig is None:
            fig, ax = plt.subplots()

        # Weight plot

        if omit_sites is not None :
            ax.scatter(xs_omit, ys_omit, **kwargs_omit)

        if color_home_cell:
            # Zip the home cell coordinates into tuples
            home_ev_coords = set(zip(xs_ev_home, ys_ev_home))

            # Filter even sites: Keep (x, y) pairs that are not in home_coordinates
            out_even = [(x, y) for x, y in zip(xs_ev, ys_ev) if (x, y) not in home_ev_coords]
            if out_even:
                xs_ev_out, ys_ev_out = zip(*out_even)
            else:
                xs_ev_out, ys_ev_out = [], []  # In case no points are left

            # Zip the home cell coordinates into tuples
            home_odd_coords = set(zip(xs_odd_home, ys_odd_home))

            # Filter even sites: Keep (x, y) pairs that are not in home_coordinates
            out_odd = [(x, y) for x, y in zip(xs_odd, ys_odd) if (x, y) not in home_odd_coords]
            if out_even:
                xs_odd_out, ys_odd_out = zip(*out_odd)
            else:
                xs_odd_out, ys_odd_out = [], []  # In case no points are left
            
            if 'c' in kwargs_lat.keys():
                kwargs_lat.pop('c')
            ax.scatter(xs_ev_out, ys_ev_out, zorder=2, c='k', **kwargs_lat)
            ax.scatter(xs_odd_out, ys_odd_out, zorder=2, facecolors='none', edgecolors='k', **kwargs_lat)

            ax.scatter(xs_ev_home, ys_ev_home, zorder=2, c='b', **kwargs_lat)
            ax.scatter(xs_odd_home, ys_odd_home,zorder=2, facecolors='none', edgecolors='r', **kwargs_lat)
        
        else:
            ax.scatter(xs_ev, ys_ev, zorder=2, **kwargs_lat)
            ax.scatter(xs_odd, ys_odd, zorder=2, facecolors='none', edgecolors='k', **kwargs_lat)

        # draw lines sectioning out home supercell
        if section_home_cell:
            c1 = np.array([0,0])
            c2 = c1 + lat_vecs[0]
            c3 = c1 + lat_vecs[1]
            c4 = c1 + lat_vecs[0] + lat_vecs[1]

            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], c='k', ls='--', lw=1)
            ax.plot([c1[0], c3[0]], [c1[1], c3[1]], c='k', ls='--', lw=1)
            ax.plot([c3[0], c4[0]], [c3[1], c4[1]], c='k', ls='--', lw=1)
            ax.plot([c2[0], c4[0]], [c2[1], c4[1]], c='k', ls='--', lw=1)

        # scatter centers
        for i in range(centers.shape[0]):
            if translate_centers:
                x = positions['centers']['xs'][i]
                y = positions['centers']['ys'][i]
                # label = fr"Center $\mathbf{{r}}^{{({i})}}_c = ({center[0]: .3f}, {center[1]: .3f})$"
                if i==0:
                    label = "Wannier centers"
                else:
                    label=None
                ax.scatter(
                    x, y, zorder=1, label=label, **kwargs_centers)
            else:
                center = centers[i]
                label = "Wannier centers"
                ax.scatter(
                    center[0], center[1], zorder=1, label=label, **kwargs_centers)

        if legend:
            ax.legend(loc='upper right')
        
        center_sc = (1/2) * (lat_vecs[0] + lat_vecs[1])
        ax.set_xlim(center_sc[0] - pmx, center_sc[0] + pmx)
        ax.set_ylim(center_sc[1] - pmy, center_sc[1] + pmy)

        ax.set_title(title)
        # ax.axis('off')

        # Saving
        if save_name is not None:
            plt.savefig(f'{save_name}.png', dpi=700)

        if show:
            plt.show()
        
        return fig, ax
    

    def plot_centers(
        self, omit_sites=None, center_scale=200,
        section_home_cell=True, color_home_cell=True, translate_centers=False,
        title=None, save_name=None, show=False, legend=False, pmx=4, pmy=4,
        kwargs_centers={'s': 80, 'marker': '*', 'c': 'g'},
        kwargs_omit={'s': 50, 'marker': 'x', 'c':'k'},
        kwargs_lat={'s':10, 'marker': 'o', 'c':'k'}, fig=None, ax=None
    ):
        lat_vecs = self.model.get_lat_vecs()
        orbs = self.model.get_orb_vecs(Cartesian=False)
        centers = self.centers
        wfs = self.WFs

        # Initialize arrays to store positions and weights
        positions = {
            'all': {'xs': [], 'ys': []},
            'centers': {'xs': [[] for i in range(centers.shape[0])], 'ys':[[] for i in range(centers.shape[0])]},
            'home even': {'xs': [], 'ys': []},
            'home odd': {'xs': [], 'ys': []},
            'omit': {'xs': [], 'ys': []},
            'even': {'xs': [], 'ys': []},
            'odd': {'xs': [], 'ys': []},
        }
        for tx, ty in self.supercell:
            if translate_centers:
                for j in range(centers.shape[0]):
                    center = centers[j] + tx * lat_vecs[0] + ty * lat_vecs[1]
                    positions['centers']['xs'][j].append(center[0])
                    positions['centers']['ys'][j].append(center[1])
            for i, orb in enumerate(orbs):
                # Extract relevant parameters
                pos = orb[0] * lat_vecs[0] + tx * lat_vecs[0] + orb[1] * lat_vecs[1] + ty * lat_vecs[1]
                x, y = pos[0], pos[1]

                # Store values in 'all'
                positions['all']['xs'].append(x)
                positions['all']['ys'].append(y)

                # Handle omit site if applicable
                if omit_sites is not None and i in omit_sites:
                    positions['omit']['xs'].append(x)
                    positions['omit']['ys'].append(y)
                # Separate even and odd index sites
                if i % 2 == 0:
                    positions['even']['xs'].append(x)
                    positions['even']['ys'].append(y)
                    if tx == ty == 0:
                        positions['home even']['xs'].append(x)
                        positions['home even']['ys'].append(y)
                else:
                    positions['odd']['xs'].append(x)
                    positions['odd']['ys'].append(y)
                    if tx == ty == 0:
                        positions['home odd']['xs'].append(x)
                        positions['home odd']['ys'].append(y)


        # Convert lists to numpy arrays (batch processing for cleanliness)
        for key, data in positions.items():
            for sub_key in data:
                positions[key][sub_key] = np.array(data[sub_key])

        # All positions
        xs = positions['all']['xs']
        ys = positions['all']['ys']

        # home cell site positions
        xs_ev_home = positions['home even']['xs']
        ys_ev_home = positions['home even']['ys']
        xs_odd_home = positions['home odd']['xs']
        ys_odd_home = positions['home odd']['ys']

        # omitted site positions
        xs_omit = positions['omit']['xs']
        ys_omit = positions['omit']['ys']

        # sublattice positions
        xs_ev = positions['even']['xs']
        ys_ev = positions['even']['ys']
        xs_odd = positions['odd']['xs']
        ys_odd = positions['odd']['ys']

        
        if fig is None:
            fig, ax = plt.subplots()

        # Weight plot

        if omit_sites is not None :
            ax.scatter(xs_omit, ys_omit, **kwargs_omit)

        if color_home_cell:
            # Zip the home cell coordinates into tuples
            home_ev_coords = set(zip(xs_ev_home, ys_ev_home))

            # Filter even sites: Keep (x, y) pairs that are not in home_coordinates
            out_even = [(x, y) for x, y in zip(xs_ev, ys_ev) if (x, y) not in home_ev_coords]
            if out_even:
                xs_ev_out, ys_ev_out = zip(*out_even)
            else:
                xs_ev_out, ys_ev_out = [], []  # In case no points are left

            # Zip the home cell coordinates into tuples
            home_odd_coords = set(zip(xs_odd_home, ys_odd_home))

            # Filter even sites: Keep (x, y) pairs that are not in home_coordinates
            out_odd = [(x, y) for x, y in zip(xs_odd, ys_odd) if (x, y) not in home_odd_coords]
            if out_even:
                xs_odd_out, ys_odd_out = zip(*out_odd)
            else:
                xs_odd_out, ys_odd_out = [], []  # In case no points are left
            
            if 'c' in kwargs_lat.keys():
                kwargs_lat.pop('c')
            ax.scatter(xs_ev_out, ys_ev_out, zorder=2, c='k', **kwargs_lat)
            ax.scatter(xs_odd_out, ys_odd_out, zorder=2, facecolors='none', edgecolors='k', **kwargs_lat)

            ax.scatter(xs_ev_home, ys_ev_home, zorder=2, c='k', **kwargs_lat)
            ax.scatter(xs_odd_home, ys_odd_home,zorder=2, facecolors='none', edgecolors='k', **kwargs_lat)
        
        else:
            ax.scatter(xs_ev, ys_ev, zorder=2, **kwargs_lat)
            ax.scatter(xs_odd, ys_odd, zorder=2, facecolors='none', edgecolors='k', **kwargs_lat)

        # draw lines sectioning out home supercell
        if section_home_cell:
            c1 = np.array([0,0])
            c2 = c1 + lat_vecs[0]
            c3 = c1 + lat_vecs[1]
            c4 = c1 + lat_vecs[0] + lat_vecs[1]

            ax.plot([c1[0], c2[0]], [c1[1], c2[1]], c='k', ls='--', lw=1)
            ax.plot([c1[0], c3[0]], [c1[1], c3[1]], c='k', ls='--', lw=1)
            ax.plot([c3[0], c4[0]], [c3[1], c4[1]], c='k', ls='--', lw=1)
            ax.plot([c2[0], c4[0]], [c2[1], c4[1]], c='k', ls='--', lw=1)

        # scatter centers
        for i in range(centers.shape[0]):
            if translate_centers:
                x = positions['centers']['xs'][i]
                y = positions['centers']['ys'][i]
                if i==0:
                    label = "Wannier centers"
                else:
                    label=None
                ax.scatter(
                    x, y, zorder=1, label=label, s=np.exp(10*self.spread[i])*center_scale, **kwargs_centers)
            else:
                center = centers[i]
                label = "Wannier centers"
                ax.scatter(
                    center[0], center[1], zorder=1, label=label, **kwargs_centers)

        if legend:
            ax.legend(loc='upper right')
        
        center_sc = (1/2) * (lat_vecs[0] + lat_vecs[1])
        ax.set_xlim(center_sc[0] - pmx, center_sc[0] + pmx)
        ax.set_ylim(center_sc[1] - pmy, center_sc[1] + pmy)

        ax.set_title(title)

        # Saving
        if save_name is not None:
            plt.savefig(f'{save_name}.png', dpi=700)

        if show:
            plt.show()
        
        return fig, ax
        