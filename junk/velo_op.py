import numpy as np

def vel(model, klist):
    '''
    Computes the derivative of H_k at a set of given k-points.
    The output derivatives are wrt Cartesian coordinates (kx,ky).
    
    :param model: the TB model.
    :param klist: list of k-points at which the derivative is evaluated.
    The input list is in reduced coordinates (k1,k2).
    '''
    dim_k = model._dim_k
    kpnts = len(klist)
    n_orb = model.get_num_orbitals()
    hoppings = model._hoppings

    # Initialize velocity matrices corresponding to the k1 and k2 derivatives of H_k
    velred_x = np.zeros((kpnts, n_orb, n_orb), dtype=complex)
    velred_y = np.zeros((kpnts, n_orb, n_orb), dtype=complex)

    # Initialize array of lattice vectors to compute the B array
    a1 = model._lat[0]
    a2 = model._lat[1]
    A = np.zeros((dim_k, dim_k))
    A[0] = a1[0: dim_k]
    A[1] = a2[0:dim_k]
    A = A/(2*np.pi)

    # B array enables chain rule to go from (k1, k2) to (kx, ky)
    B = np.transpose(A) # Used later for chain rule

    for m in range(kpnts):
        for hopping in hoppings:
            #See _gen_ham function
            kpnt = klist[m]
            amp = complex(hopping[0])

            i = hopping[1]
            j = hopping[2]

            ind_R = np.array(hopping[3], dtype=float)

            rv = -model._orb[i, :] + model._orb[j, :] + ind_R
            rv = rv[model._per]

            phase = np.exp((2.0j) * np.pi * np.dot(kpnt, rv))

            amp_x = 2j * np.pi * rv[0] * amp * phase # Derivative of H_k wrt k1
            amp_y = 2j * np.pi * rv[1] * amp * phase # Derivative of H_k wrt k2
            # print(f"{i}, {j}, k = {kpnt}, ampx = {amp_x}, ampy = {amp_y}")

            velred_x[m, i, j] += amp_x
            velred_x[m, j, i] += amp_x.conjugate()
            velred_y[m, i, j] += amp_y
            velred_y[m, j, i] += amp_y.conjugate()

    #Apply chain rule to obtain velocity matrix in Cartesian basis
    vel_x = B[0][0] * velred_x + B[0][1] * velred_y
    vel_y = B[1][0] * velred_x + B[1][1] * velred_y

    vel = np.concatenate((vel_x[:, np.newaxis, ...], vel_y[:, np.newaxis, ...]), axis=1)

    return vel