import copy
from itertools import product, permutations
from itertools import combinations_with_replacement as comb
from functools import partial
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pythtb import tb_model, wf_array
from math import factorial


# def get_periodic_H(model, H_flat, k_vals):
#     orb_vecs = model.get_orb_vecs()
#     orb_vec_diff = orb_vecs[:, None, :] - orb_vecs[None, :, :]
#     # orb_phase = np.exp(1j * 2 * np.pi * np.einsum('ijm, ...m->...ij', orb_vec_diff, k_vals))
#     orb_phase = np.exp(1j * 2 * np.pi * np.matmul(orb_vec_diff, k_vals.T)).transpose(2,0,1)
#     H_per_flat = H_flat * orb_phase
#     return H_per_flat


# def vel_op_fin_diff(model, H_flat, k_vals, dk, order_eps=1, mode='central'):
#     """
#     Compute velocity operators using finite differences.
    
#     Parameters:
#         H_mesh: ndarray of shape (Nk, M, M)
#             The Hamiltonian on the parameter grid.
#         dk: list of float
#             Step sizes in each parameter direction.
    
#     Returns:
#         v_mu_fd: list of ndarray
#             Velocity operators for each parameter direction.
#     # """

#     # recip_lat_vecs = model.get_recip_lat_vecs()
#     # recip_basis = recip_lat_vecs/ np.linalg.norm(recip_lat_vecs, axis=1, keepdims=True)
#     # g = recip_basis @ recip_basis.T
#     # sqrt_mtrc = np.sqrt(np.linalg.det(g))
#     # g_inv = np.linalg.inv(g)

#     # dk = np.einsum("ij, j -> i", g_inv, dk)

#     # assume only k for now
#     dim_param = model._dim_k # Number of parameters (dimensions)
#     # assume equal number of mesh points along each dimension
#     nks = ( int(H_flat.shape[0]**(1/dim_param)),)*dim_param

#     # Switch to periodic gauge H(k) = H(k+G) 
#     H_flat = get_periodic_H(model, H_flat, k_vals)
#     H_mesh = H_flat.reshape(*nks, model._norb, model._norb)
#     v_mu_fd = np.zeros((dim_param, *H_mesh.shape), dtype=complex)

#     # Compute Jacobian
#     recip_lat_vecs = model.get_recip_lat_vecs()
#     inv_recip_lat = np.linalg.inv(recip_lat_vecs)
 
#     for mu in range(dim_param):
#         coeffs, stencil = finite_diff_coeffs(order_eps=order_eps, mode=mode)

#         derivative_sum = np.zeros_like(H_mesh)

#         for s, c in zip(stencil, coeffs):
#             H_shifted = np.roll(H_mesh, shift=-s, axis=mu)
#             derivative_sum += c * H_shifted

#         v_mu_fd[mu] = derivative_sum / (dk[mu])

#         # Ensure Hermitian symmetry
#         v_mu_fd[mu] = 0.5 * (v_mu_fd[mu] + np.conj(v_mu_fd[mu].swapaxes(-1, -2)))

#     return v_mu_fd


def finite_diff_coeffs(order_eps, derivative_order=1, mode='central'):
    """
    Compute finite difference coefficients using the inverse of the Vandermonde matrix.

    Parameters:
        stencil_points (array-like): The relative positions of the stencil points (e.g., [-2, -1, 0, 1, 2]).
        derivative_order (int): Order of the derivative to approximate (default is first derivative).

    Returns:
        coeffs (numpy array): Finite difference coefficients for the given stencil.
    """
    if mode not in ["central", "forward", "backward"]:
        raise ValueError("Mode must be 'central', 'forward', or 'backward'.")
    
    num_points = derivative_order + order_eps  

    if mode == "central":
        if num_points % 2 == 0:
            num_points += 1
        half_span = num_points//2
        stencil = np.arange(-half_span, half_span + 1)

    elif mode == "forward":
        stencil = np.arange(0, num_points)

    elif mode == "backward":
        stencil = np.arange(-num_points+1, 1)

    A = np.vander(stencil, increasing=True).T  # Vandermonde matrix
    b = np.zeros(num_points)
    b[derivative_order] = factorial(derivative_order) # Right-hand side for the desired derivative

    coeffs = np.linalg.solve(A, b)  # Solve system Ax = b
    return coeffs, stencil


def compute_d4k_and_d2k(delta_k):
    """
    Computes the 4D volume element d^4k and the 2D plaquette areas d^2k for a given set of difference vectors in 4D space.

    Parameters:
    delta_k (numpy.ndarray): A 4x4 matrix where each row is a 4D difference vector.

    Returns:
    tuple: (d4k, plaquette_areas) where
        - d4k is the absolute determinant of delta_k (4D volume element).
        - plaquette_areas is a dictionary with keys (i, j) and values representing d^2k_{ij}.
    """
    # Compute d^4k as the determinant of the 4x4 difference matrix
    d4k = np.abs(np.linalg.det(delta_k))

    # Function to compute 2D plaquette area in 4D space
    def compute_plaquette_area(v1, v2):
        """Compute the 2D plaquette area spanned by two 4D vectors."""
        area_squared = 0.0
        # Sum over all unique (m, n) pairs where m < n
        for m in range(4):
            for n in range(m + 1, 4):
                area_squared += (v1[m] * v2[n] - v1[n] * v2[m]) ** 2
        return np.sqrt(area_squared)

    # Compute all unique plaquette areas
    plaquette_areas = {}
    for i in range(4):
        for j in range(i + 1, 4):
            plaquette_areas[(i, j)] = compute_plaquette_area(delta_k[i], delta_k[j])

    return d4k, plaquette_areas


def levi_civita(n, d):
    """
    Constructs the rank-n Levi-Civita tensor in dimension d.

    Parameters:
    n (int): Rank of the tensor (number of indices).
    d (int): Dimension (number of possible index values).

    Returns:
    np.ndarray: Levi-Civita tensor of shape (d, d, ..., d) with n dimensions.
    """
    shape = (d,) * n
    epsilon = np.zeros(shape, dtype=int)
    # Generate all possible permutations of n indices
    for perm in permutations(range(d), n):
        # Compute the sign of the permutation
        sign = np.linalg.det(np.eye(n)[list(perm)])
        epsilon[perm] = int(np.sign(sign))  # +1 for even, -1 for odd permutations

    return epsilon


class Model(tb_model):
    def __init__(self, dim_k, dim_r, lat=None, orb=None, per=None, nspin=1):
        super().__init__(dim_k, dim_r, lat=lat, orb=orb, per=per, nspin=nspin)

        self.dim_k = dim_k
        self.dim_r = dim_r
        self.n_spin = nspin
        self.n_orb = super().get_num_orbitals()
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
        """
        Generate Bloch Hamiltonian for an array of k-points and varying parameters

        This Hamiltonian follows tight-binding convention I where the phase factors
        associated with the orbital positions are included. This means H(k) =\= H(k+G), but 
        instead H(k) = U H(k+G) U^(dagger). Taking finite differences for partial k_mu H(k)
        doesn't work in convention I at the boundaries.

        Args:
            k_pts (array-like, optional): Array of k-points in reduced coordinates
        """
        
        k_arr = np.array(k_pts) if k_pts is not None else None

        if k_pts is not None:
            # if kpnt is just a number then convert it to an array
            if len(k_arr.shape) == 0:
                k_arr = np.array([k_arr])
            
            n_kpts = k_arr.shape[0]

            # check that k-vector is of corect size
            if k_arr.shape[-1] != self._dim_k:
                raise Exception("\n\nk-vector of wrong shape!")
        else:
            if self.dim_k != 0:
                raise Exception("\n\nHave to provide a k-vector!")

        if self._nspin == 1:
            ham = np.zeros((n_kpts, self._norb, self._norb), dtype=complex)
        elif self._nspin == 2:
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
                if self._nspin == 2:        
                    # Compute the amplitude for all k-points and components
                    amp = phase[:, None, None] * amp  # Shape: (n_kpts, n_spin, n_spin)
                else: 
                    amp *= phase  # Shape: (n_kpts,)

            if self._nspin == 1:
                ham[..., i, j] += amp 
                ham[..., j, i] += amp.conjugate()
            elif self._nspin == 2:
                ham[..., i, :, j, :] += amp
                ham[..., j, :, i, :] += np.swapaxes(amp.conjugate(), -1, -2)

        return ham
    
    
    def get_periodic_H(self, H_flat, k_vals):
        """
        Change to periodic gauge so that H(k+G) = H(k)

        If n_spin = 2, H_flat should only be flat along k and NOT spin
        """
        orb_vecs = self.get_orb_vecs()
        orb_vec_diff = orb_vecs[:, None, :] - orb_vecs[None, :, :]
        # np.matmul(orb_vec_diff, k_vals.T) = np.einsum('ijm, ...m->...ij', orb_vec_diff, k_vals)).T
        # np.matmul(orb_vec_diff, k_vals.T) has shape (n_orb, n_orb, N_k)
        orb_phase = np.exp(1j * 2 * np.pi * np.matmul(orb_vec_diff, k_vals.T)).transpose(2,0,1)
        H_per_flat = H_flat * orb_phase
        return H_per_flat
    
    
    def solve_ham(self, k_arr, return_eigvecs=False):
        """
        Returns eigenergies and eigenvectors along k_path

        Returns:
            eigvals (Nk, n_orb*n_spin)
            eigvecs (Nk, n_orb*n_spin, n_orb [, n_spin])
        """
        H_k = self.get_ham(k_arr) # (Nk, n_orb, n_orb) or (Nk, n_orb, n_spin, n_orb, n_spin)

        # flatten spin
        new_shape = (H_k.shape[0],) + (self._nspin*self._norb, self._nspin*self._norb)
        H_k = H_k.reshape(*new_shape) 

        # H_k is shape (Nk, n_orb*n_spin, n_orb*n_spin)

        if np.max(H_k-np.swapaxes(H_k.conj(), -1, -2)) > 1e-9:
            raise Exception("\n\nHamiltonian matrix is not hermitian?!")
        
        if not return_eigvecs:
            evals = np.linalg.eigvalsh(H_k)
            return evals
        else:
            eigvals, eigvecs = np.linalg.eigh(H_k)
            eigvecs = np.swapaxes(eigvecs, -1, -2)  # (Nk, n_orb*n_spin, n_orb*n_spin)

            if self._nspin == 2:
                new_shape = (H_k.shape[0],) + (self._nsta, self._norb, 2)
                eigvecs = eigvecs.reshape(*new_shape) # (Nk, n_orb*n_spin, n_orb, n_spin]

            return eigvals, eigvecs
    

    def gen_velocity(self, k_pts):
        """
        Generate the velocity operator using commutator v_k = i[H_k, r] for an array of k-points.
        
        Parameters:
            model: Tight-binding model instance.
            k_pts: Array of k-points in reduced coordinates, shape (n_kpts, dim_k).
        
        Returns:
            vel: Velocity operators at each k-point, shape (dim_k, n_kpts, n_orb, n_orb).
        """
        
        k_pts = np.array(k_pts)
        n_kpts, dim_k = k_pts.shape

        if dim_k != self._dim_k:
            raise ValueError("k-points have incorrect dimension.")

        n_orb = self._norb

        if self._nspin == 1:
            vel = np.zeros((dim_k, n_kpts, n_orb, n_orb), dtype=complex)
        elif self._nspin == 2:
            vel = np.zeros((dim_k, n_kpts, n_orb, 2, n_orb, 2), dtype=complex)
        else:
            raise ValueError("Invalid spin value.")

        # Precompute the lattice vectors for periodic directions
        # lat_per: lattice vectors in Cartesian coordinates for the periodic directions.
        lat_per = self.get_lat()[self._per, :]  # Shape: (dim_k, dim_k)
        # recip_lat: full reciprocal lattice vectors (each is a vector)
        recip_lat = self.get_recip_lat_vecs()
        # For non-orthogonal lattices, we need the full inverse matrix.
        inv_recip_lat = np.linalg.inv(recip_lat)
        recip_lat_mag = [np.linalg.norm(vec) for vec in recip_lat]

        # Loop over all hoppings
        for hopping in self._hoppings:
            if self._nspin == 1:
                amp = complex(hopping[0])
            elif self._nspin == 2:
                amp = np.array(hopping[0], dtype=complex)

            i = hopping[1]
            j = hopping[2]
            ind_R = np.array(hopping[3], dtype=float)

            # Compute the displacement in real space (including orbital offsets)
            delta_r = ind_R + self._orb[j, :] - self._orb[i, :]  # Shape: (dim_r,)
            # Keep only the periodic (reduced) components
            delta_r_per = delta_r[self._per]  # Shape: (dim_k,)

            # Compute phase factors for all k-points
            phase = np.exp(1j * 2 * np.pi * k_pts @ delta_r_per)  # Shape: (n_kpts,)
            # Include Jacobian for non-orthogonal lattices
            deriv =  1j * 2*np.pi * (inv_recip_lat @ delta_r_per)

            # Compute the amplitude for all k-points and components
            if self._nspin == 2:
                # Compute the amplitude for all k-points and components
                amp_k = (
                    deriv[:, np.newaxis, np.newaxis, np.newaxis] * 
                    phase[np.newaxis, :, np.newaxis, np.newaxis] * 
                    amp[np.newaxis, np.newaxis, :, :] 
                    )
                  # Shape: (dim_k, n_kpts, n_spin, n_spin)
            else: 
                amp_k =  amp * deriv[:, np.newaxis] *  phase[np.newaxis, :] # Shape: (dim_k, n_kpts)

            if self._nspin == 1:
                # Update velocity operator
                vel[..., i, j] += amp_k 
                vel[..., j, i] += np.conj(amp_k)
            elif self._nspin == 2:
                vel[..., i, :, j, :] += amp_k
                vel[..., j, :, i, :] += np.swapaxes(amp_k.conjugate(), -1, -2)

        return vel
    

    # def quantum_geom_tens(self, k_pts, occ_idxs=None):
    #     dim_k = k_pts.shape[-1]
    #     Nk = k_pts.shape[0]
        
    #     evals, evecs = self.solve_ham(k_pts, return_eigvecs=True)
    #     n_eigs = evecs.shape[1]

    #     if occ_idxs is None:
    #         n_occ = int(n_eigs/2)
    #         occ_idxs = list(range(n_occ))

    #     n_occ = len(occ_idxs)

    #     v_k = self.gen_velocity(k_pts) # shape (dim_k, n_kpts, n_orb, n_orb)
    #     v_k_rot = np.einsum("...ni, a...ij, ...mj -> ...anm", evecs.conj(), v_k, evecs)  # (n_kpts, dim_k, n_orb, n_orb)
    #     delta_E = evals[..., np.newaxis, :] - evals[..., :, np.newaxis]
    #     delta_E_sq = delta_E**2
    #     QGT = np.zeros((Nk, dim_k, dim_k, len(occ_idxs)), dtype=complex)

    #     for n in range(n_occ):
    #         for m in range(n_eigs):
    #             if (n in occ_idxs) ^ (m in occ_idxs):
    #                 for mu in range(dim_k):
    #                     v_nm_mu = v_k_rot[:, mu, n, m]
    #                     for nu in range(dim_k):
    #                         v_mn_nu = v_k_rot[:, nu, m, n]
    #                         QGT[:, mu, nu, n] += (v_nm_mu * v_mn_nu) / delta_E_sq[:, n, m]

    #     return QGT

    
    def berry_curvature(self, k_pts, occ_idxs=None, periodic_gauge=False):

        H_flat = self.get_ham(k_pts) # (Nk, n_orb, n_orb) or (Nk, n_orb, n_spin, n_orb, n_spin)
        v_k = self.gen_velocity(k_pts) # (Nk, dim_k, n_orb, n_orb)

        if periodic_gauge:
            # should have spin axis separate
            H_flat = self.get_periodic_H(H_flat, k_pts)

        # flatten spin
        new_shape = (H_flat.shape[0],) + (self._nspin*self._norb, self._nspin*self._norb)
        H_flat = H_flat.reshape(*new_shape) 

        # H_flat is shape (Nk, n_orb*n_spin, n_orb*n_spin)

        if np.max(H_flat-np.swapaxes(H_flat.conj(), -1, -2)) > 1e-9:
            raise Exception("\n\nHamiltonian matrix is not hermitian?!")
        
        evals, evecs = np.linalg.eigh(H_flat)
        # swap for consistent indexing: eigval index is -2, and vals are -1
        evecs = np.swapaxes(evecs, -1, -2)  # (Nk, n_orb*n_spin, n_orb*n_spin)
        # TODO: for spinful calculations, should we do this?
        # new_shape = H_flat.shape[0] + (self._nsta, self._norb, self._nspin) 
        # evecs = evecs.reshape(*new_shape) # (Nk, n_orb*n_spin, n_orb, n_spin]

        n_eigs = evecs.shape[-2]

        # Identify occupied bands
        if occ_idxs is None:
            occ_idxs =  np.arange(n_eigs//2)
        else:
            occ_idxs = np.array(occ_idxs)

        # Identify conduction bands
        cond_idxs = np.setdiff1d(np.arange(n_eigs), occ_idxs)  # Identify conduction bands

        # Compute energy denominators in vectorized way
        delta_E = evals[..., np.newaxis, :] - evals[..., :, np.newaxis] # shape (Nk, n_states, n_states)
        with np.errstate(divide="ignore", invalid="ignore"):  # Suppress warnings
            inv_delta_E = np.where(delta_E != 0, 1 / delta_E, 0)

        # Rotate velocity operators to eigenbasis
        evecs_conj = evecs.conj()[np.newaxis, :, : :]
        evecs_T =  evecs.transpose(0,2,1)[ np.newaxis, :, : :]
        vk_evecT = np.matmul(v_k, evecs_T)
        v_k_rot = np.matmul(evecs_conj, vk_evecT) # (dim_k, n_kpts, n_orb, n_orb)

        # Extract relevant submatrices
        v_occ_cond = v_k_rot[..., occ_idxs, :][..., :, cond_idxs] # shape (dim_k, Nk, n_occ, n_con)
        v_cond_occ = v_k_rot[..., cond_idxs, :][..., :, occ_idxs] # shape (dim_k, Nk, n_con, n_occ)
        delta_E_occ_cond = inv_delta_E[:, occ_idxs, :][:, :, cond_idxs] # shape (Nk, n_con, n_occ)

        v_occ_cond = v_occ_cond * delta_E_occ_cond
        v_cond_occ = v_cond_occ * delta_E_occ_cond.swapaxes(-1,-2)

        # Berry curvature shape: (dim_k, dim_k, n_kpts, n_orb, n_orb)
        b_curv = 1j * ( 
            np.matmul(v_occ_cond[:, None], v_cond_occ[None, :]) 
            - np.matmul(v_occ_cond[None, :], v_cond_occ[:, None])
        )

        return b_curv
        

    def Chern(self, dirs=(0,1)):
        """
        Only works for 2d systems right now
        """
        
        nks = (100,) * self._dim_k
        k_mesh = K_mesh(self, *nks)
        flat_mesh = k_mesh.gen_k_mesh(endpoint=False)
        Omega = self.berry_curvature(flat_mesh)

        Nk = Omega.shape[2] 
        V_BZ = self.get_recip_vol()
        dk_sq = V_BZ / Nk

        Chern = np.sum(np.trace(Omega[dirs], axis1=-1, axis2=-2)) * dk_sq / (2 * np.pi)

        return Chern.real
        
    
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
        self, k_path, nk=101, k_label=None, proj_orb_idx=None, 
        proj_spin=False, fig=None, ax=None, title=None, scat_size=3, 
        lw=2, lc='b', ls='solid', cmap="bwr", show=False, cbar=True
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
        

        # scattered bands with sublattice color
        if proj_orb_idx is not None:
            # diagonalize model on path
            evals, evecs = self.solve_ham(k_vec, return_eigvecs=True)
            n_eigs = evals.shape[-1]

            if self._nspin == 1:
                wt = abs(evecs)**2
                col = np.sum([wt[..., i] for i in proj_orb_idx], axis=0)
            elif self._nspin == 2:
                wt = abs(evecs)**2
                col = np.sum([wt[..., i, :] for i in proj_orb_idx], axis=(0,-1))

            for n in range(n_eigs):
                scat = ax.scatter(k_dist, evals[:, n], c=col[:, n], cmap=cmap, marker='o', s=scat_size, vmin=0, vmax=1, zorder=2)

            if cbar:
                cbar = fig.colorbar(scat, ticks=[1,0], pad=0.01)
                # cbar.set_ticks([])
                # cbar.ax.set_yticklabels([r'$B$', r'$A$'], size=12)
                cbar.ax.set_yticklabels([r'$\psi_B$', r'$\psi_A$'], size=12)

        elif proj_spin:
            evals, evecs = self.solve_ham(k_vec, return_eigvecs=True)
            n_eigs = evals.shape[-1]

            assert self._nspin != 1, "Spin needs to be greater than 1."

            wt = abs(evecs)**2
            col = np.sum(wt[..., 1], axis=2)

            for n in range(n_eigs):
                scat = ax.scatter(k_dist, evals[:, n], c=col[:, n], cmap=cmap, marker='o', s=scat_size, vmin=0, vmax=1, zorder=2)

            cbar = fig.colorbar(scat, ticks=[1,0])
            cbar.ax.set_yticklabels(['spin up', 'spin down'], size=12)

        else:
            evals = self.solve_ham(k_vec, return_eigvecs=False)
            n_eigs = evals.shape[-1]

            # continuous bands
            for n in range(n_eigs):
                ax.plot(k_dist, evals[:, n], c=lc, lw=lw, ls=ls)

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
        self.Nk = np.prod(nks)
        self.dim: int = len(nks)
        self.recip_lat_vecs = model.get_recip_lat_vecs()
        self.idx_arr: list = list(product(*[range(nk) for nk in nks]))  # 1D list of all k_indices (integers)
        self.square_mesh: np.ndarray = self.gen_k_mesh(flat=False, endpoint=False) # each index is a direction in k-space
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
        flat_mesh = np.array(list(product(*k_vals)))

        return flat_mesh if flat else flat_mesh.reshape(*[nk for nk in self.nks], len(self.nks))
    
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
        nnbr_idx = list(product(list(range(-N_sh, N_sh + 1)), repeat=self.dim))
        nnbr_idx.remove((0,)*self.dim)
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
        if self.model._nspin == 1:
            bc_phase = np.ones((*self.nks, idx_shell[0].shape[0], self.model._orb_vecs.shape[0]), dtype=complex)
        else:
            bc_phase = np.ones((*self.nks, idx_shell[0].shape[0], self.model._orb_vecs.shape[0], 2), dtype=complex)

        for k_idx in self.idx_arr: # each index in k-mesh
            for shell_idx, idx_vec in enumerate(idx_shell[0]):  # vecs connecting neighboring indices
                k_nbr_idx = np.array(k_idx) + idx_vec # indices for neighboring k
                mod_idx = np.mod(k_nbr_idx, self.nks) # apply pbc to index
                diff = k_nbr_idx - mod_idx # if mod changed nbr, will be non-zero
                G = np.divide(np.array(diff), np.array(self.nks)) # will be pm 1 where crossed, 0 else
                # if the translated k-index contains -1 or nk_i+1 then we crossed the BZ boundary
                cross_bndry = np.any((k_nbr_idx == -1) | np.logical_or.reduce([k_nbr_idx == nk for nk in self.nks]))
                if cross_bndry:
                    if self.model._nspin == 1:
                        bc_phase[k_idx][shell_idx] = np.exp(-1j * 2 * np.pi * self.model._orb_vecs @ G.T).T
                    else:
                        bc_phase[k_idx][shell_idx, :] = np.exp(-1j * 2 * np.pi * self.model._orb_vecs @ G.T).T[..., np.newaxis]

        if self.model._nspin == 2:
            bc_phase = bc_phase.reshape(*bc_phase.shape[:self.dim+1], -1)
                    

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


class Bloch(wf_array):
    def __init__(self, model: Model, *param_dims):
        """Class for storing and manipulating Bloch like wavefunctions.
        
        Wavefunctions are defined on a semi-full reciprocal space mesh.
        """
        super().__init__(model, param_dims)
        assert len(param_dims) >= model._dim_k, "Number of dimensions must be >= number of reciprocal space dimensions"

        # total parameter space dimension
        self.dim_param = len(param_dims)

        # reciprocal space dimensions
        self.dim_k = model._dim_k
        self.nks = param_dims[:self.dim_k]
        # stores k-points on a uniform mesh, calculates nearest neighbor points given the model lattice
        self.k_mesh: K_mesh = K_mesh(model, *self.nks)

        # adiabatic dimension
        self.dim_lam = self.dim_param - self.dim_k 
        self.n_lambda = param_dims[self.dim_k:]

        # periodic boundary conditions assumed false unless specified
        self.pbc_lam = False

        # model attributes
        self.model: Model = model
        self._n_orb = model.get_num_orbitals()
        self._nspin = self.model._nspin
        self._n_states = self._n_orb * self._nspin

        # axes indexes
        self.k_axes = tuple(range(self.dim_k))
        self.lambda_axes = tuple(range(self.dim_k, self.dim_param))

        if self._nspin == 2:
            self.spin_axis = -1
            self.orb_axis = -2
            self.state_axis = -3
        else:
            self.spin_axis = None
            self.orb_axis = -1
            self.state_axis = -2
        
        # wavefunction shapes
        if self.dim_lam > 0:
            if self._nspin == 2:
                self._wf_shape = (*self.nks, *self.n_lambda, self._n_states, self._n_orb, self._nspin)
            else:
                self._wf_shape = (*self.nks, *self.n_lambda, self._n_states, self._n_orb)
        else:
            if self._nspin == 2:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb, self._nspin)
            else:
                self._wf_shape = (*self.nks, self._n_states, self._n_orb)

            
        self.set_Bloch_ham()

    def get_wf_axes(self):
        dict_axes = {
            "wf shape": self._wf_shape,
            "Number of axes": len(self._wf_shape),
            "k-axes": self.k_axes, "lambda-axes": self.lambda_axes, "spin-axis": self.spin_axis,
            "orbital axis": self.orb_axis, "state axis": self.state_axis
            }
        return dict_axes

    def set_pbc_lam(self):
        self.pbc_lam = True


    def set_Bloch_ham(self, lambda_vals=None, model_fxn=None):
        if lambda_vals is None:
            H_k = self.model.get_ham(k_pts=self.k_mesh.flat_mesh) # [Nk, norb, norb]
            # [nk1, nk2, ..., norb, norb]
            self.H_k = H_k.reshape(*[nk for nk in self.k_mesh.nks], *H_k.shape[1:])
            return
        
        lambda_keys = list(lambda_vals.keys())
        lambda_ranges = list(lambda_vals.values())
        lambda_shape = tuple(len(vals) for vals in lambda_ranges)
        dim_lambda = len(lambda_keys)

        n_kpts = self.k_mesh.Nk
        n_orb = self._n_orb
        n_spin = self._n_spin
        n_states = n_orb*n_spin

        # Initialize storage for wavefunctions and energies
        if n_spin == 1:
            H_kl = np.zeros((*lambda_shape, n_kpts, n_states, n_states), dtype=complex)
        elif n_spin == 2:
            H_kl = np.zeros((*lambda_shape, n_kpts, n_orb, n_spin, n_orb, n_spin), dtype=complex)

        for idx, param_set in enumerate(np.ndindex(*lambda_shape)):
            # kwargs for model_fxn with specified parameter values
            param_dict = {lambda_keys[i]: lambda_ranges[i][param_set] for i in range(dim_lambda)}

            # Generate the model with modified parameters
            modified_model = model_fxn(**param_dict)

            H_kl[param_set] = modified_model.get_ham(k_pts=self.k_mesh.flat_mesh)

        # Reshape for compatibility with existing Berry curvature methods
       
        if self._nspin == 1:
            new_axes = (dim_lambda,) + tuple(range(dim_lambda)) + tuple(range(dim_lambda+1, dim_lambda+3))
        else:
            new_axes = (dim_lambda,) + tuple(range(dim_lambda))+tuple(range(dim_lambda+1, dim_lambda+5))
        H_kl = np.transpose(H_kl, axes=new_axes)

        if self._nspin == 1:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_states)
        else:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_orb, n_spin)
        H_kl = H_kl.reshape(new_shape)
        
        self.H_k = H_kl


    def solve_model(self, model_fxn=None, lambda_vals=None):
        """
        Solves for the eigenstates of the Bloch Hamiltonian defined by the model over a semi-full 
        k-mesh, e.g. in 3D reduced coordinates {k = [kx, ky, kz] | k_i in [0, 1)}.

        Args:
            model_fxn (function, optional):
                A function that returns a model given a set of parameters.
            param_vals (dict, optional):
                Dictionary of parameter values for adiabatic evoltuion. Each key corresponds to
                a varying parameter and the values are arrays
        """

        if lambda_vals is None:
            # compute eigenstates and eigenenergies on full k_mesh
            eigvals, eigvecs = self.model.solve_ham(self.k_mesh.flat_mesh, return_eigvecs=True)
            eigvecs = eigvecs.reshape(*self.k_mesh.nks, *eigvecs.shape[1:])
            eigvals = eigvals.reshape(*self.k_mesh.nks, *eigvals.shape[1:])
            self.set_wfs(eigvecs)
            self.energies = eigvals
            return

        lambda_keys = list(lambda_vals.keys())
        lambda_ranges = list(lambda_vals.values())
        lambda_shape = tuple(len(vals) for vals in lambda_ranges)
        dim_lambda = len(lambda_keys)
        
        n_kpts = self.k_mesh.Nk
        n_orb = self.model.get_num_orbitals()
        n_spin = self.model.n_spin
        n_states = n_orb*n_spin

        # Initialize storage for wavefunctions and energies
        if n_spin == 1:
            u_wfs = np.zeros((*lambda_shape, n_kpts, n_states, n_states), dtype=complex)
        elif n_spin == 2:
            u_wfs = np.zeros((*lambda_shape, n_kpts, n_states, n_orb, n_spin), dtype=complex)

        energies = np.zeros((*lambda_shape, n_kpts, n_states))

        for idx, param_set in enumerate(np.ndindex(*lambda_shape)):
            param_dict = {lambda_keys[i]: lambda_ranges[i][param_set] for i in range(dim_lambda)}

            # Generate the model with modified parameters
            modified_model = model_fxn(**param_dict)
             
            # Solve for eigenstates
            eigvals, eigvecs = modified_model.solve_ham(self.k_mesh.flat_mesh, return_eigvecs=True)

            # Store results
            energies[param_set] = eigvals
            u_wfs[param_set] = eigvecs

        # Reshape for compatibility with existing Berry curvature methods
        new_axes = (dim_lambda,) + tuple(range(dim_lambda))+(dim_lambda+1, )
        energies = np.transpose(energies, axes=new_axes)
        if self._nspin == 1:
            new_axes = (dim_lambda,) + tuple(range(dim_lambda))+tuple(range(dim_lambda+1, dim_lambda+3))
        else:
            new_axes = (dim_lambda,) + tuple(range(dim_lambda))+tuple(range(dim_lambda+1, dim_lambda+4))
        u_wfs = np.transpose(u_wfs, axes=new_axes)

        if self._nspin == 1:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_states)
        else:
            new_shape = (*self.k_mesh.nks, *lambda_shape, n_states, n_orb, n_spin)
        u_wfs = u_wfs.reshape(new_shape)
        energies = energies.reshape((*self.k_mesh.nks, *lambda_shape, n_states))

        self.set_wfs(u_wfs, cell_periodic=True)
        self.energies = energies

    def solve_on_path(self, k_arr):
        """
        Solves on model passed when initialized. Not suitable for 
        adiabatic parameters in the model beyond k.
        """
        eigvals, eigvecs = self.model.solve_ham(k_arr, return_eigvecs=True)
        self.set_wfs(eigvecs)
        self.energies = eigvals
        

    ###### Retrievers  #######

    def get_states(self, flatten_spin=False):
        """Returns dictionary containing Bloch and cell-periodic eigenstates."""
        assert hasattr(self, "_psi_wfs"), "Need to call `solve_model` or `set_wfs` to initialize Bloch states"
        psi_wfs = self._psi_wfs
        u_wfs = self._u_wfs

        if flatten_spin:
            # shape is [nk1, ..., nkd, [n_lambda,] n_state, n_orb, n_spin], flatten last two axes
            psi_wfs = psi_wfs.reshape((*psi_wfs.shape[:-2], -1))
            u_wfs = u_wfs.reshape((*u_wfs.shape[:-2], -1))

        return {"Bloch": psi_wfs,  "Cell periodic": u_wfs}
    
    
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
            self._psi_wfs = self._apply_phase(wfs)
        else:
            self._psi_wfs = wfs
            self._u_wfs = self._apply_phase(wfs, inverse=True)

        if self._nspin != 2:
            # overlap matrix
            self._M = self._get_self_overlap_mat()
            # band projectors
            self._set_projectors()


    def _get_pbc_wfs(self):

        dim_k = self.k_mesh.dim
        orb_vecs = self.model.get_orb_vecs(Cartesian=False)

        # Initialize the extended array by padding with an extra element along each k-axis
        pbc_uwfs = np.pad(
            self._u_wfs, pad_width=[(0, 1) if i < dim_k else (0, 0) for i in range(self._u_wfs.ndim)], mode="wrap")
        pbc_psiwfs = np.pad(
            self._psi_wfs, pad_width=[(0, 1) if i < dim_k else (0, 0) for i in range(self._psi_wfs.ndim)], mode="wrap")

        # Compute the reciprocal lattice vectors (unit vectors for each dimension)
        G_vectors = list(product([0, 1], repeat=dim_k))
        # Remove the zero vector
        G_vectors = [np.array(vector) for vector in G_vectors if any(vector)]
        
        for  G in G_vectors:
            phase = np.exp(-1j * 2 * np.pi * (orb_vecs @ G.T)).T[np.newaxis, :]
            slices_new = []
            slices_old = []

            for i, value in enumerate(G):
                if value == 1:
                    slices_new.append(slice(-1, None))  # Take the last element along this axis
                    slices_old.append(slice(0, None))
                else:
                    slices_new.append(slice(None))  # Take all elements along this axis
                    slices_old.append(slice(None))  # Take all elements along this axis

            # Add slices for any remaining dimensions (m, n) if necessary
            slices_new.extend([slice(None)] * (pbc_uwfs.ndim - len(G)))
            slices_old.extend([slice(None)] * (pbc_uwfs.ndim - len(G)))
            pbc_uwfs[tuple(slices_new)] *= phase

        return pbc_psiwfs, pbc_uwfs
    
    # Works with and without spin and lambda
    def _apply_phase(self, wfs, inverse=False):
        """
        Change between cell periodic and Bloch wfs by multiplying exp(\pm i k . tau)

        Args:
        wfs (pythtb.wf_array): Bloch or cell periodic wfs [k, nband, norb]

        Returns:
        wfsxphase (np.ndarray): 
            wfs with orbitals multiplied by phase factor

        """
        lam = -1 if inverse else 1  # overall minus if getting cell periodic from Bloch
        per_dir = list(range(self.k_mesh.flat_mesh.shape[-1]))  # list of periodic dimensions
        # slice second dimension to only keep only periodic dimensions in orb
        per_orb = self.model._orb_vecs[:, per_dir]

        # compute a list of phase factors: exp(pm i k . tau) of shape [k_val, orbital]
        phases = np.exp(lam * 1j * 2 * np.pi * per_orb @ self.k_mesh.flat_mesh.T, dtype=complex).T
        phases = phases.reshape(*self.k_mesh.nks, self._n_orb)

        if hasattr(self, "n_lambda") and self.n_lambda:
            phases = phases[..., np.newaxis, :]

        wfs = wfs.reshape(*self._wf_shape)
        
        # broadcasting to match dimensions
        if self._nspin == 1:
            # reshape to have each k-dimension as an axis
            # wfs = wfs.reshape(*self.k_mesh.nks, self._n_states, self._n_orb)
            # newaxis along state dimension
            phases = phases[..., np.newaxis, :]
        elif self._nspin == 2:
            # reshape to have each k-dimension as an axis
            # newaxis along state and spin dimension
            phases = phases[..., np.newaxis, :, np.newaxis]

        return wfs * phases
    
    # TODO: allow for projectors onto subbands
    # TODO: possibly get rid of nbr by storing boundary states
    def _set_projectors(self):
        num_nnbrs = self.k_mesh.num_nnbrs
        nnbr_idx_shell = self.k_mesh.nnbr_idx_shell

        if self._nspin == 2:
            u_wfs = self.get_states(flatten_spin=True)["Cell periodic"]
        else:
            u_wfs = self.get_states()["Cell periodic"]

        # band projectors
        self._P = np.einsum("...ni, ...nj -> ...ij", u_wfs, u_wfs.conj())
        self._Q = np.eye(self._n_orb*self._nspin) - self._P

        self._P_nbr = np.zeros((self._P.shape[:-2] + (num_nnbrs,) + self._P.shape[-2:]), dtype=complex)
        self._Q_nbr = np.zeros_like(self._P_nbr)
        
        # self._P_nbr = np.zeros((*nks, num_nnbrs, self._n_orb*self._nspin, self._n_orb*self._nspin), dtype=complex)
        # self._Q_nbr = np.zeros((*nks, num_nnbrs, self._n_orb*self._nspin, self._n_orb*self._nspin), dtype=complex)

        #TODO need shell to iterate over extra lambda dims also, shift accordingly
        for idx, idx_vec in enumerate(nnbr_idx_shell[0]):  # nearest neighbors
            # accounting for phase across the BZ boundary
            states_pbc = np.roll(
                u_wfs, shift=tuple(-idx_vec), axis=self.k_axes
                ) * self.k_mesh.bc_phase[..., idx, np.newaxis,  :]
            self._P_nbr[..., idx, :, :] = np.einsum(
                    "...ni, ...nj -> ...ij", states_pbc, states_pbc.conj()
                    )
            self._Q_nbr[..., idx, :, :] = np.eye(self._n_orb*self._nspin) - self._P_nbr[..., idx, :, :]

        return
    
    # TODO: allow for subbands
    def _get_self_overlap_mat(self):
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
        _, idx_shell = self.k_mesh.get_k_shell(N_sh=1, report=False)
        idx_shell = idx_shell[0]
        bc_phase = self.k_mesh.bc_phase

        # overlap matrix
        M = np.zeros(
            (*self.k_mesh.nks, len(idx_shell), self._n_states, self._n_states), dtype=complex
        )  

        if self._nspin == 2:
            u_wfs = self.get_states(flatten_spin=True)["Cell periodic"]
        else:
            u_wfs = self.get_states()["Cell periodic"]

        print(u_wfs.shape)       
        for idx, idx_vec in enumerate(idx_shell):  # nearest neighbors
            # introduce phases to states when k+b is across the BZ boundary
            states_pbc = np.roll(
                u_wfs, shift=tuple(-idx_vec), axis=[i for i in range(self.k_mesh.dim)]
                ) * bc_phase[..., idx, np.newaxis,  :]
            M[..., idx, :, :] = np.einsum("...mj, ...nj -> ...mn", u_wfs.conj(), states_pbc)
            
        return M

    def berry_phase(self, dir=0, state_idx=None, evals=False):
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
        if state_idx is not None:
            wfs = np.take(wfs, state_idx, axis=self.state_axis)
        orb_vecs = self.model.get_orb_vecs()
        dim_param = self.k_mesh.dim  # dimensionality of parameter space
        param_axes = np.arange(0, dim_param)  # parameter axes
        param_axes = np.setdiff1d(param_axes, dir)  # remove dir from axes to loop
        lens = [wfs.shape[ax] for ax in param_axes]  # sizes of loop directions
        idxs = np.ndindex(*lens)  # index mesh
        
        phase = np.zeros((*lens, wfs.shape[dim_param]))

        G = np.zeros(dim_param)
        G[0] = 1
        phase_shift = np.exp(-1j * 2 * np.pi * (orb_vecs @ G.T))
        print(param_axes)
        for idx_set in idxs:
            # print(idx_set)
            # take wfs along loop axis at given idex
            sliced_wf = wfs.copy()
            for ax, idx in enumerate(idx_set):
                # print(param_axes[ax])
                sliced_wf = np.take(sliced_wf, idx, axis=param_axes[ax])

            # print(sliced_wf.shape)
            end_state = sliced_wf[0,...] * phase_shift[np.newaxis, :, np.newaxis]
            sliced_wf = np.append(sliced_wf, end_state[np.newaxis, ...], axis=0)
            phases = self.berry_loop(sliced_wf, evals=evals)
            phase[idx_set] = phases

        return phase        

    # works in all cases
    def wilson_loop(self, wfs_loop, evals=False):
        """Compute Wilson loop unitary matrix and its eigenvalues for multiband Berry phases.
        
        Multiband Berry phases always returns numbers between -pi and pi.  
        
        Args:
            wfs_loop (np.ndarray):
                Has format [loop_idx, band, orbital, spin] and loop has to be one dimensional.
                Assumes that first and last loop-point are the same. Therefore if
                there are n wavefunctions in total, will calculate phase along n-1
                links only!  
            berry_evals (bool):
                If berry_evals is True then will compute phases for
                individual states, these corresponds to 1d hybrid Wannier
                function centers. Otherwise just return one number, Berry phase.
        """

        # number of occupied states
        nocc = wfs_loop.shape[1]
        # temporary matrices
        wfs_loop = wfs_loop.reshape(wfs_loop.shape[0], wfs_loop.shape[1], -1)

        U_wilson = np.eye(nocc, dtype=complex)

        ovr_mats = wfs_loop[:-1].conj() @ wfs_loop[1:].swapaxes(-2, -1)

        for ovr in ovr_mats:
            V, _, Wd = np.linalg.svd(ovr, full_matrices=False)

            # update Wilson loop eigenvalues
            U_wilson @= V @ Wd


        # # go over all pairs of k-points, assuming that last point is overcounted!
        # for i in range(wfs_loop.shape[0]-1):
        #     # generate overlap matrix, go over all bands
        #     ovr = wfs_loop[i].conj() @ wfs_loop[i+1].T
    
        #     V, _, Wd = np.linalg.svd(ovr, full_matrices=False)

        #     # update Wilson loop eigenvalues
        #     U_wilson @= V @ Wd
       
        # calculate phases of all eigenvalues
        if evals:
            evals = np.linalg.eigvals(U_wilson) # Wilson loop eigenvalues
            eval_pha = -np.angle(evals) # Multiband  Berrry phases 
            return U_wilson, eval_pha
        else:
            return U_wilson
        
    # works in all cases
    def berry_loop(self, wfs_loop, evals=False):
        U_wilson = self.wilson_loop(wfs_loop, evals=evals)

        if evals:
            return U_wilson[1]
        else:
            return -np.angle(np.linalg.det(U_wilson)) # total Berry phase        

    
    def berry_flux_plaq(self, state_idx=None, non_abelian=False):
        """Compute fluxes on a two-dimensional plane of states.
        
        For a given set of states, returns the band summed Berry curvature
        rank-2 tensor for all combinations of surfaces in reciprocal space. 
        By convention, the Berry curvature is reported at the point where the loop
        started, which is the lower left corner of a plaquette. 
        """

        wfs = self.get_states()["Cell periodic"] # u_nk's
        orb_vecs = self.model._orb_vecs

        if state_idx is not None:
            wfs = np.take(wfs, state_idx, axis=self.state_axis)
            # need extra axis for single state
            if isinstance(state_idx, int):
                wfs = wfs[:self.state_axis-1, np.newaxis, ...]

        # size of the mesh
        nks = self.k_mesh.nks
        n_lambda = self.n_lambda
        dim = self.dim_param
        n_states = wfs.shape[self.state_axis]

        # store flux through each plaquette of the mesh
        if non_abelian:
            if self.dim_lam > 0:
                Berry_flux = np.zeros((dim, dim, *nks, *n_lambda, n_states, n_states), dtype=complex)
            else:
                Berry_flux = np.zeros((dim, dim, *nks, n_states, n_states), dtype=complex)
        else:
            if self.dim_lam > 0:
                Berry_flux = np.zeros((dim, dim, *nks, *n_lambda), dtype=complex)
            else:
                Berry_flux = np.zeros((dim, dim, *nks), dtype=complex)

        if self.dim_lam > 0:
            adia_idxes = list(np.ndindex(*nks, *n_lambda))
        else:
            adia_idxes = list(np.ndindex(*nks))

        n_lam_max = [n-1 for n in n_lambda]

        # loop over unique surfaces ij: (xy, xz, yz, etc.)
        for i in range(dim): 
            for j in range(i + 1, dim):
                print(i,j)
                if non_abelian:
                    if self.dim_lam > 0:
                        phases_plane = np.zeros((*nks, *n_lambda, n_states, n_states), dtype=complex)
                    else:
                        phases_plane = np.zeros((*nks, n_states, n_states), dtype=complex)
                else:
                    if self.dim_lam > 0:
                        phases_plane = np.zeros((*nks, *n_lambda), dtype=complex)
                    else:
                        phases_plane = np.zeros(nks, dtype=complex)
                
                # loop over all plaquettes in this plane
                for idx in adia_idxes:
                    idx = np.array(idx) # e.g. for 3D [0, 0, 0] ... [3, 0, 0] ... [3, 5, 0] etc.

                    if not self.pbc_lam and self.dim_lam > 0:
                        if (np.any(idx[self.k_mesh.dim:] == n_lam_max) and (i == self.k_mesh.dim or j == self.k_mesh.dim) ):
                            print("here")

                            continue

                    # define indices of the 4 points of the plaquette, add to the index if in the plane
                    # e.g. idx = [1, 0, 0], (i=1, j=2) --> computing Omega_yz, so we have 
                    # [ [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 0, 0] ]
                    plaquette_points = [
                        idx,
                        idx + np.array([1 if k == i else 0 for k in range(dim)]),
                        idx + np.array([1 if k == i else 0 for k in range(dim)])
                            + np.array([1 if k == j else 0 for k in range(dim)]),
                        idx + np.array([1 if k == j else 0 for k in range(dim)]),
                        idx
                    ]

                    # Extract wavefunctions for the loop
                    wf_use = []
                    for p_idx in plaquette_points:
                        # correctly finds idx with pbc, assuming pbc on all indices
                        # print(p_idx)
                        if self.dim_lam > 0:
                            if self.pbc_lam:
                                mod_idx = np.mod(p_idx, (*nks, *n_lambda))  # Apply PBC to index
                            else:
                                mod_idx_k = np.mod(p_idx[:self.k_mesh.dim], nks)
                                mod_idx_lam = p_idx[self.k_mesh.dim:]
                                mod_idx = (*mod_idx_k, *mod_idx_lam)
                        else:
                            mod_idx = np.mod(p_idx, nks)

                        # print(mod_idx)
                        # print()

                        wf_nbr = np.copy(wfs[tuple(mod_idx)])  # (n_states, n_orb, [opt: n_spin])
                        # print(wf_nbr.shape)

                        diff = p_idx - mod_idx  # Nonzero if crossed BZ boundary 
                        diff_k = diff[:self.k_mesh.dim]
                        crossed_BZ_bndry = np.any(diff_k)
                        # print(crossed_BZ_bndry)

                        if crossed_BZ_bndry:  # Apply boundary condition for u_nk
                            G = np.divide(diff_k, nks)  # G-vector is 1 along dimension crossed
                            # print(G)
                            # print(orb_vecs.T)
                            # print(G @ orb_vecs.T)
                            phase = np.exp(-2j * np.pi * G @ orb_vecs.T) # (n_orb,)

                            if self._nspin == 1:
                                # newaxis for states
                                phase =  phase[np.newaxis, ...] # (1, n_orb) 
                            elif self._nspin == 2:
                                # newaxis for states and spin
                                phase = phase[np.newaxis, ..., np.newaxis] # (1, n_orb, 1)
                            
                            wf_nbr = wf_nbr * phase
                        
                        wf_use.append(wf_nbr)
                        # print()

                    wf_use = np.array(wf_use, dtype=complex)

                    if non_abelian:
                        U_wilson = self.wilson_loop(wf_use)
                        eigvals, eigvecs = np.linalg.eig(U_wilson)
                        phases_plane[tuple(idx)] = eigvecs @ -np.angle(np.diag(eigvals)) @ np.linalg.inv(eigvecs)

                    else:
                        # Compute Berry phase for this plaquette
                        phases_plane[tuple(idx)] = self.berry_loop(wf_use)
        
                # Store the phases for this plane
                # TODO: Is Non-Abelian still antisymmetric?
                Berry_flux[i, j] = phases_plane
                Berry_flux[j, i] = -phases_plane

        return Berry_flux
            
    #TODO finish non_Abelian
    def berry_curv(self, dirs=None, state_idx=None, non_abelian=False, diff_lam=1, return_flux=False):
        nks = self.nks
        recip_lat_vecs = self.model.get_recip_lat_vecs()

        # Get number of spatial (k-space) and adiabatic (λ-space) dimensions
        dim_k = len(self.nks)      # Number of k-space dimensions
        dim_lam = len(self.n_lambda)  # Number of adiabatic dimensions
        dim_total = dim_k + dim_lam          # Total number of dimensions

        if self.dim_lam > 0:
            # Construct delta_k with 4D embedding (extra zeros for adiabatic params)
            delta_k = np.zeros((dim_k, dim_total))  # (3, dim_total)
            delta_k[:, :dim_k] = recip_lat_vecs / np.array(self.nks)[:, None]

            # Construct delta_lambda (embedding into 4D with zeros in k-space parts)
            delta_lam = np.zeros((dim_lam, dim_total))  # (dim_lam, dim_total)
            np.fill_diagonal(delta_lam[:, dim_k:], diff_lam / np.array(self.n_lambda))

            # Stack k-space and adiabatic-space vectors together
            dk = np.vstack([delta_k, delta_lam])  # (dim_total, dim_total)
            print(dk)
            
        else:
            dk = np.array([recip_lat_vecs[k_idx]/(nk) for k_idx, nk in enumerate(nks)])
            print(dk)
        
        # Occupied states
        Berry_flux = self.berry_flux_plaq(state_idx=state_idx, non_abelian=non_abelian) 
        Berry_curv = np.zeros_like(Berry_flux, dtype=complex)
        dim = Berry_flux.shape[0] # number of dimensions in parameter space
        # Loop over planes (off-diagonal elements only)
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    # Skip diagonal elements; these should remain zero
                    continue
                else:
                    # Area element for the (i, j)-plane
                    A = np.vstack([dk[i], dk[j]])
                    # area_element = np.prod([np.linalg.norm(dk[i]), np.linalg.norm(dk[j])])
                    area_element = np.sqrt(np.linalg.det(A @ A.T))
                    print(A)
                    # area_element = np.linalg.det(A)
                    
                    print(area_element)
       
                    # Divide flux by the area element to get approx curvature
                    Berry_curv[i, j] = Berry_flux[i, j] / area_element

        if dirs is not None:
            if return_flux:
                return Berry_curv[dirs], Berry_flux[dirs]
            else:
                return Berry_curv[dirs]
        else:
            if return_flux:
                return Berry_curv, Berry_flux
            else:
                return Berry_curv
    

    def chern_num(self, dirs=(0,1), band_idxs=None):
        if band_idxs is None:
            n_occ = int(self._n_states/2)
            band_idxs = np.arange(n_occ) # assume half-filled occupied

        berry_flux = self.berry_flux_plaq(state_idx=band_idxs)
        Chern = np.sum(berry_flux[dirs]/(2*np.pi))

        return Chern
    
    # TODO allow for subbands
    def trace_metric(self):
        P = self._P
        Q_nbr = self._Q_nbr

        nks = Q_nbr.shape[:-3]
        num_nnbrs = Q_nbr.shape[-3]
        w_b, _, _ = self.k_mesh.get_weights(N_sh=1)

        T_kb = np.zeros((*nks, num_nnbrs), dtype=complex)
        for nbr_idx in range(num_nnbrs):  # nearest neighbors
            T_kb[..., nbr_idx] = np.trace(P[..., :, :] @ Q_nbr[..., nbr_idx, :, :], axis1=-1, axis2=-2)

        return w_b[0] * np.sum(T_kb, axis=-1)
    
    # #TODO allow for subbands
    # def omega_til(self):
    #     M = self._M
    #     w_b, k_shell, idx_shell = self.k_mesh.get_weights(N_sh=1)
    #     w_b = w_b[0]
    #     k_shell = k_shell[0]

    #     nks = M.shape[:-3]
    #     Nk = np.prod(nks)
    #     k_axes = tuple([i for i in range(len(nks))])

    #     diag_M = np.diagonal(M, axis1=-1, axis2=-2)
    #     log_diag_M_imag = np.log(diag_M).imag
    #     abs_diag_M_sq = abs(diag_M) ** 2

    #     r_n = -(1 / Nk) * w_b * np.sum(log_diag_M_imag, axis=k_axes).T @ k_shell

    #     Omega_tilde = (1 / Nk) * w_b * ( 
    #             np.sum((-log_diag_M_imag - k_shell @ r_n.T)**2) + 
    #             np.sum(abs(M)**2) - np.sum(abs_diag_M_sq)
    #         )
    #     return Omega_tilde
    

    def interp_op(self, O_k, k_path, plaq=False):
        k_mesh = np.copy(self.k_mesh.square_mesh)
        k_idx_arr = self.k_mesh.idx_arr
        nks = self.k_mesh.nks
        dim_k = len(nks)
        Nk = np.prod([nks])

        supercell = list(product(*[range(-int((nk-nk%2)/2), int((nk-nk%2)/2)) for nk in nks]))

        if plaq:
            # shift by half a mesh point to get the center of the plaquette
            k_mesh += np.array([(1/nk)/2 for nk in nks])

        # Fourier transform to real space
        O_R = np.zeros((len(supercell), *O_k.shape[dim_k:]), dtype=complex)
        for idx, pos in enumerate(supercell):
            for k_idx in k_idx_arr:
                R_vec = np.array(pos)
                phase = np.exp(-1j * 2 * np.pi * np.vdot(k_mesh[k_idx], R_vec))
                O_R[idx] += O_k[k_idx] * phase / Nk

        # interpolate to arbitrary k
        O_k_interp = np.zeros((k_path.shape[0], *O_k.shape[dim_k:]), dtype=complex)
        for k_idx, k in enumerate(k_path):
            for idx, pos in enumerate(supercell):
                R_vec = np.array(pos)
                phase = np.exp(1j * 2 * np.pi * np.vdot(k, R_vec))
                O_k_interp[k_idx] += O_R[idx] * phase

        return O_k_interp
    

    def interp_energy(self, k_path, return_eigvecs=False):
        H_k_proj = self.get_proj_ham()
        H_k_interp = self.interp_op(H_k_proj, k_path)
        if return_eigvecs:
            u_k_interp = self.interp_op(self._u_wfs, k_path)
            eigvals_interp, eigvecs_interp = np.linalg.eigh(H_k_interp)
            eigvecs_interp = np.einsum("...ij, ...ik -> ...jk", u_k_interp, eigvecs_interp)
            eigvecs_interp = np.transpose(eigvecs_interp, axes=[0, 2, 1])
            return eigvals_interp, eigvecs_interp
        else:
            eigvals_interp = np.linalg.eigvalsh(H_k_interp)
            return eigvals_interp
    
    #TODO allow for subbands
    def get_proj_ham(self):
        H_k_proj = self._u_wfs.conj() @ self.H_k @ np.swapaxes(self._u_wfs, -1, -2)
        return H_k_proj

    #TODO allow for subbands
    def plot_interp_bands(
        self, k_path, nk=101, k_label=None, red_lat_idx=None, 
        fig=None, ax=None, title=None, scat_size=3, 
        lw=2, lc='b', ls='solid', cmap="bwr", show=False, cbar=True
        ):
        if fig is None and ax is None:
            fig, ax = plt.subplots()

        (k_vec, k_dist, k_node) = self.model.k_path(k_path, nk, report=False)
        k_vec = np.array(k_vec)

        if red_lat_idx is not None:
            eigvals, eigvecs = self.interp_energy(k_vec, return_eigvecs=True)

            n_eigs = eigvecs.shape[-2]
            wt = abs(eigvecs)**2
            col = np.sum([  wt[..., i] for i in red_lat_idx ], axis=0)

            for n in range(n_eigs):
                scat = ax.scatter(k_dist, eigvals[:, n], c=col[:, n], cmap=cmap, marker='o', s=scat_size, vmin=0, vmax=1, zorder=2)

            if cbar:
                cbar = fig.colorbar(scat, ticks=[1,0])
                cbar.ax.set_yticklabels([r'$\psi_2$', r'$\psi_1$'], size=12)

        else:
            eigvals = self.interp_energy(k_vec)

            # continuous bands
            for n in range(eigvals.shape[1]):
                ax.plot(k_dist, eigvals[:, n], c=lc, lw=lw, ls=ls)

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

        

class Wannier():
    def __init__(
            self, model: Model, nks: list  
            ):
        self.model: Model = model
        self._nks: list = nks
        self.k_mesh: K_mesh = K_mesh(model, *nks)

        self.energy_eigstates: Bloch = Bloch(model, *nks)
        self.energy_eigstates.solve_model()
        self.tilde_states: Bloch = Bloch(model, *nks)

        self.supercell = list(product(*[range(-int((nk-nk%2)/2), int((nk-nk%2)/2)) for nk in nks]))  # used for real space looping of WFs
    
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
            tf_list: list[int | list[tuple]]
                list of tuples defining the orbital and amplitude of the trial function
                on that orbital. Of the form [ [(orb, amp), ...], ...]. If spin is included,
                then the form is [ [(orb, spin, amp), ...], ...]
    
        Returns:
            tfs: np.ndarray
                Array of trial functions
        """

        # number of trial functions to define
        num_tf = len(tf_list)

        if self.model._nspin == 2:
            tfs = np.zeros([num_tf, self.model._norb, 2], dtype=complex)
            for j, tf in enumerate(tf_list):
                assert isinstance(tf, (list, np.ndarray)), "Trial function must be a list of tuples"
                for orb, spin, amp in tf:
                    tfs[j, orb, spin] = amp
                tfs[j] /= np.linalg.norm(tfs[j])

        elif self.model._nspin == 1:
            # initialize array containing tfs = "trial functions"
            tfs = np.zeros([num_tf, self.model._norb], dtype=complex)
            for j, tf in enumerate(tf_list):
                assert isinstance(tf, (list, np.ndarray)), "Trial function must be a list of tuples"
                for site, amp in tf:
                    tfs[j, site] = amp
                tfs[j] /= np.linalg.norm(tfs[j])
            
        return tfs 
    
    def set_trial_wfs(self, tf_list):
        tfs = self.get_trial_wfs(tf_list)
        self.trial_wfs = tfs
        self.n_twfs = tfs.shape[0]
        return

    def get_tf_ovlp_mat(self, band_idxs, psi_wfs=None):
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
        if psi_wfs is None:
            # get Bloch psi_nk energy eigenstates
            psi_wfs = self.energy_eigstates.get_states()["Bloch"]

        # flatten along spin dimension in case spin is considered
        n_spin = self.model._nspin
        dim_k = self.k_mesh.dim
        num_axes = len(psi_wfs.shape)
        if num_axes != dim_k + 2 + n_spin - 1:
            # we have psi_wf defined on a 1D path in dim_k BZ
            new_shape = (*psi_wfs.shape[:2], -1)
        else:
            new_shape = (*psi_wfs.shape[:self.k_mesh.dim+1], -1)
        psi_wfs = psi_wfs.reshape(*new_shape)

        # only keep band_idxs
        psi_wfs = np.take(psi_wfs, band_idxs, axis=-2)

        assert hasattr(self, 'trial_wfs'), "Must initialize trial wfs with set_trial_wfs()"
        trial_wfs = self.trial_wfs
        # flatten along spin dimension in case spin is considered
        trial_wfs = trial_wfs.reshape((*trial_wfs.shape[:1], -1))

        A_k = np.einsum("...ij, kj -> ...ik", psi_wfs.conj(), trial_wfs)
        return A_k
    
    def set_tf_ovlp_mat(self, band_idxs):
        A_k = self.get_tf_ovlp_mat(band_idxs)
        self.A_k = A_k
        return
    
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
    
    def get_psi_tilde(self, psi_wfs, state_idx):
        """
        Performs optimal alignment of psi_wfs with tfs.
        """
        A_k = self.get_tf_ovlp_mat(state_idx, psi_wfs=psi_wfs)
        V_k, _, Wh_k = np.linalg.svd(A_k, full_matrices=False)

        # flatten spin dimensions
        psi_wfs = psi_wfs.reshape((*psi_wfs.shape[:self.k_mesh.dim+1], -1))
        # take only state_idxs
        psi_wfs = np.take(psi_wfs, state_idx, axis=-2)
        # optimal alignment
        psi_tilde = np.einsum("...mn, ...mj -> ...nj", V_k @ Wh_k, psi_wfs) # shape: (*nks, states, orbs*n_spin])

        return psi_tilde
    
    
    def single_shot(self, tf_list: list | None = None, band_idxs: list | None = None, tilde=False):
        """
        Sets the Wannier functions in home unit cell with associated spreads, centers, trial functions 
        and Bloch-like states using the single shot projection method.

        Args:
            tf_list (list): List of tuples with sites and weights. Can be un-normalized. 
            band_idxs (list | None): Band indices to Wannierize. Defaults to occupied bands (lower half).
        Returns:
            w_0n (np.array): Wannier functions in home unit cell
        """
        if tf_list is None:
            assert hasattr(self, 'trial_wfs'), "Must initialize trial wfs with set_trial_wfs()"
        else:
            self.set_trial_wfs(tf_list)

        if tilde:
            # projecting back onto tilde states
            if band_idxs is None:  # assume we are projecting onto all tilde states
                band_idxs = list(range(self.tilde_states._n_states))

            psi_til_til = self.get_psi_tilde(
                self.tilde_states._psi_wfs, state_idx=band_idxs
                )
            self.set_tilde_states(psi_til_til, cell_periodic=False)

        else:
            # projecting onto Bloch energy eigenstates
            if band_idxs is None:  # assume we are Wannierizing occupied bands
                if self.model._nspin == 1:
                    n_occ = int(self.energy_eigstates._n_states / 2)  # assuming half filled
                elif self.model._nspin == 2:
                    n_occ = int(self.energy_eigstates._n_states / 2)*2  # assuming half filled

                band_idxs = list(range(0, n_occ))

            psi_tilde = self.get_psi_tilde(self.energy_eigstates._psi_wfs, state_idx=band_idxs)
            if self.model._nspin == 2:
                psi_tilde = psi_tilde.reshape((*psi_tilde.shape[:self.k_mesh.dim+1], -1, 2))
            self.tilde_states.set_wfs(psi_tilde, cell_periodic=False)

        psi_wfs = self.tilde_states._psi_wfs
        dim_k = self.k_mesh.dim
        # DFT
        self.WFs = np.fft.ifftn(psi_wfs, axes=[i for i in range(dim_k)], norm=None)

        spread = self.spread_recip(decomp=True)
        self.spread = spread[0][0]
        self.omega_i = spread[0][1]
        self.omega_til = spread[0][2]
        self.centers = spread[1]


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
        w_b, k_shell, _ = self.k_mesh.get_weights()
        w_b, k_shell = w_b[0], k_shell[0] # Assume only one shell for now

        n_states = self.tilde_states._n_states
        nks = self.tilde_states.k_mesh.nks
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
        nks = self.tilde_states.k_mesh.nks
        Nk = self.tilde_states.k_mesh.Nk
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
        Nk = self.tilde_states.k_mesh.Nk
        n_states = self.tilde_states._n_states
        Omega_i = w_b * n_states * k_shell.shape[0] - (1 / Nk) * w_b * np.sum(abs(M) **2)
        return Omega_i
    
    
    def get_Omega_I(self, tilde=True):
        if tilde:
            P, Q = self.tilde_states.get_projector(return_Q=True)
            _, Q_nbr = self.tilde_states.get_nbr_projector(return_Q=True)
        else:
            P, Q = self.energy_eigstates.get_projector(return_Q=True)
            _, Q_nbr = self.energy_eigstates.get_nbr_projector(return_Q=True)

        nks = self.k_mesh.nks
        Nk = np.prod(nks)
        num_nnbrs = self.k_mesh.num_nnbrs
        w_b, _, idx_shell = self.k_mesh.get_weights(N_sh=1)

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
    
        nks = self.k_mesh.nks
        Nk = np.prod(nks)
        num_nnbrs = self.k_mesh.num_nnbrs
        w_b, _, idx_shell = self.k_mesh.get_weights(N_sh=1)

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
        if self.model._nspin == 2:
            n_occ *= 2


        # eigenenergies and eigenstates for inner/outer window
        energies = self.energy_eigstates.get_energies()
        unk_states = self.energy_eigstates.get_states()["Cell periodic"]
        # initial subspace
        init_states = self.tilde_states

        if self.model._nspin == 2:
            unk_states = unk_states.reshape((*unk_states.shape[:self.k_mesh.dim+1], -1))
      
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
        w_b, _, idx_shell = self.k_mesh.get_weights(N_sh=1)
        num_nnbrs = self.k_mesh.num_nnbrs
        bc_phase = self.k_mesh.bc_phase
        
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
        w_b, _, idx_shell = self.k_mesh.get_weights(N_sh=1)
        bc_phase = self.k_mesh.bc_phase

        # initial subspace
        init_states = self.tilde_states
        energy_eigstates = self.energy_eigstates
        u_wfs = energy_eigstates.get_states(flatten_spin=True)["Cell periodic"]
        # u_wfs_til = init_states.get_states(flatten_spin=True)["Cell periodic"]

        if N_wfs is None:
            # assume number of states in the subspace is number of tilde states 
            N_wfs = init_states._n_states

        if outer_bands == "occupied":
            outer_bands = list(range(n_occ))

        outer_states = u_wfs.take(outer_bands, axis=-2)

        # Projector of initial tilde subspace at each k-point
        if inner_bands is None:
            N_inner = 0
            P = init_states.get_projector()
            P_nbr, Q_nbr = init_states.get_nbr_projector(return_Q=True)
            T_kb = np.einsum('...ij, ...kji -> ...k', P, Q_nbr)
        else:
            N_inner = len(inner_bands)
            inner_states = u_wfs.take(inner_bands, axis=-2)

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
        comp_states = u_wfs.take(comp_bands, axis=-2)

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
                Q_nbr_min = np.eye(n_orb*self.model._nspin) - P_nbr_min 
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
        w_b, k_shell, idx_shell = self.k_mesh.get_weights()
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
                    np.swapaxes(U, -1, -2).conj() @  M0[..., idx, :, :] @ np.roll(U, shift=tuple(-idx_vec), axis=(0,1))
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
            
            if grad_mag_prev < grad_mag and i != 0:
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
        
        u_tilde_wfs = self.tilde_states.get_states(flatten_spin=True)["Cell periodic"]
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
            # if self.model._nspin == 2:
            #     u_tilde = self.tilde_states.get_states(flatten_spin=True)["Cell periodic"]
            # else:
            u_tilde = self.tilde_states.get_states(flatten_spin=False)["Cell periodic"]
        if wan_idxs is not None:
            u_tilde = np.take_along_axis(u_tilde, wan_idxs, axis=-2)

        H_k = self.get_Bloch_Ham()
        if self.model._nspin == 2:
            new_shape = H_k.shape[:-4] + (2*self.model._norb, 2*self.model._norb)
            H_k = H_k.reshape(*new_shape)

        H_rot_k = u_tilde.conj() @ H_k @ np.swapaxes(u_tilde, -1, -2)
        eigvals, eigvecs = np.linalg.eigh(H_rot_k)
        eigvecs = np.einsum('...ij, ...ik->...kj', u_tilde, eigvecs)
        # eigvecs = np.swapaxes(eigvecs, -1, -2)

        k_mesh = self.k_mesh.square_mesh
        k_idx_arr = self.k_mesh.idx_arr
        nks = self.k_mesh.nks
        Nk = np.prod([nks])

        supercell = list(product(*[range(-int((nk-nk%2)/2), int((nk-nk%2)/2)+1) for nk in nks]))

        # Fourier transform to real space
        H_R = np.zeros((len(supercell), H_rot_k.shape[-1], H_rot_k.shape[-1]), dtype=complex)
        u_R = np.zeros((len(supercell), u_tilde.shape[-2], u_tilde.shape[-1]), dtype=complex)
        eval_R = np.zeros((len(supercell), eigvals.shape[-1]), dtype=complex)
        evecs_R = np.zeros((len(supercell), eigvecs.shape[-2], eigvecs.shape[-1]), dtype=complex)
        for idx, r in enumerate(supercell):
            for k_idx in k_idx_arr:
                R_vec = np.array([*r])
                phase = np.exp(-1j * 2 * np.pi * np.vdot(k_mesh[k_idx], R_vec))
                # H_R[idx, :, :] += H_rot_k[k_idx] * phase / Nk
                # u_R[idx] += u_tilde[k_idx] * phase / Nk
                eval_R[idx] += eigvals[k_idx] * phase / Nk
                evecs_R[idx] += eigvecs[k_idx] * phase / Nk

        # interpolate to arbitrary k
        H_k_interp = np.zeros((k_path.shape[0], H_R.shape[-1], H_R.shape[-1]), dtype=complex)
        u_k_interp = np.zeros((k_path.shape[0], u_R.shape[-2], u_R.shape[-1]), dtype=complex)
        eigvals_k_interp = np.zeros((k_path.shape[0], eval_R.shape[-1]), dtype=complex)
        eigvecs_k_interp = np.zeros((k_path.shape[0], evecs_R.shape[-2], evecs_R.shape[-1]), dtype=complex)

        for k_idx, k in enumerate(k_path):
            for idx, r in enumerate(supercell):
                R_vec = np.array([*r])
                phase = np.exp(1j * 2 * np.pi * np.vdot(k, R_vec))
                # H_k_interp[k_idx] += H_R[idx] * phase
                # u_k_interp[k_idx] += u_R[idx] * phase
                eigvals_k_interp[k_idx] += eval_R[idx] * phase
                eigvecs_k_interp[k_idx] += evecs_R[idx] * phase

        # eigvals, eigvecs = np.linalg.eigh(H_k_interp)
        # eigvecs = np.einsum('...ij, ...ik -> ...kj', u_k_interp, eigvecs)
        # # normalizing
        # eigvecs /= np.linalg.norm(eigvecs, axis=-1, keepdims=True)
        eigvecs_k_interp /= np.linalg.norm(eigvecs_k_interp, axis=-1, keepdims=True)

        if ret_eigvecs:
            return eigvals_k_interp.real, eigvecs_k_interp
        else:
            return eigvals
    
    
    def interp_op(self, O_k, k_path, plaq=False):
        return self.tilde_states.interp_op(O_k, k_path, plaq=plaq)
       

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
        w0 = self.WFs#.reshape((*self.WFs.shape[:self.k_mesh.dim+1], -1))
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
                wt = np.sum(np.abs(wf_value) ** 2)
                # phase = np.arctan2(wf_value.imag, wf_value.real) 
                pos = orb[0] * lat_vecs[0] + tx * lat_vecs[0] + orb[1] * lat_vecs[1] + ty * lat_vecs[1]
                rel_pos = pos - center
                x, y, rad = pos[0], pos[1], np.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2)

                # Store values in 'all'
                positions['all']['xs'].append(x)
                positions['all']['ys'].append(y)
                positions['all']['r'].append(rad)
                positions['all']['wt'].append(wt)
                # positions['all']['phase'].append(phase)

                # Handle omit site if applicable
                if omit_sites is not None and i in omit_sites:
                    positions['omit']['xs'].append(x)
                    positions['omit']['ys'].append(y)
                    positions['omit']['r'].append(rad)
                    positions['omit']['wt'].append(wt)
                    # positions['omit']['phase'].append(phase)
                # Separate even and odd index sites
                if i % 2 == 0:
                    positions['even']['xs'].append(x)
                    positions['even']['ys'].append(y)
                    positions['even']['r'].append(rad)
                    positions['even']['wt'].append(wt)
                    # positions['even']['phase'].append(phase)
                    if tx == ty == 0:
                        positions['home even']['xs'].append(x)
                        positions['home even']['ys'].append(y)
                        positions['home even']['r'].append(rad)
                        positions['home even']['wt'].append(wt)
                        # positions['home even']['phase'].append(phase)

                else:
                    positions['odd']['xs'].append(x)
                    positions['odd']['ys'].append(y)
                    positions['odd']['r'].append(rad)
                    positions['odd']['wt'].append(wt)
                    # positions['odd']['phase'].append(phase)
                    if tx == ty == 0:
                        positions['home odd']['xs'].append(x)
                        positions['home odd']['ys'].append(y)
                        positions['home odd']['r'].append(rad)
                        positions['home odd']['wt'].append(wt)
                        # positions['home odd']['phase'].append(phase)


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
        #     fig2, ax2 = plst.subplots()
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
        self, omit_sites=None, center_scale=200,
        section_home_cell=True, color_home_cell=True, translate_centers=False,
        title=None, save_name=None, show=False, legend=False, pmx=4, pmy=4,
        kwargs_centers={'s': 80, 'marker': '*', 'c': 'g'},
        kwargs_omit={'s': 50, 'marker': 'x', 'c':'k'},
        kwargs_lat_ev={'s':10, 'marker': 'o', 'c':'k'}, 
        kwargs_lat_odd={'s':10, 'marker': 'o', 'facecolors':'none', 'edgecolors':'k'},
        fig=None, ax=None
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

            ax.scatter(xs_ev_home, ys_ev_home, zorder=2, **kwargs_lat_ev)
            ax.scatter(xs_odd_home, ys_odd_home, zorder=2, **kwargs_lat_odd)
            
            if 'c' in kwargs_lat_ev.keys():
                kwargs_lat_ev.pop('c')
            if 'c' in kwargs_lat_odd.keys():
                kwargs_lat_odd.pop('c')

            ax.scatter(xs_ev_out, ys_ev_out, zorder=2, **kwargs_lat_ev)
            ax.scatter(xs_odd_out, ys_odd_out, zorder=2, **kwargs_lat_odd)
        
        else:
            ax.scatter(xs_ev, ys_ev, zorder=2, **kwargs_lat_ev)
            ax.scatter(xs_odd, ys_odd, zorder=2, **kwargs_lat_odd)

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
                    x, y, zorder=1, label=label, s=np.exp(11*self.spread[i])*center_scale, **kwargs_centers)
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
    

    def plot_interp_bands(self, k_path, nk=101, k_label=None, red_lat_idx=None, 
        fig=None, ax=None, title=None, scat_size=3, 
        lw=2, lc='b', ls='solid', cmap="plasma", show=False, cbar=True):

        return self.tilde_states.plot_interp_bands(
            k_path, nk=nk, k_label=k_label, red_lat_idx=red_lat_idx, fig=fig, ax=ax,
            title=title, scat_size=scat_size, lw=lw, lc=lc, ls=ls, cmap=cmap, show=show, cbar=cbar
            )
        