from pythtb import *

# used for testing purposes

def chessboard(t0, tprime, delta):
    # define lattice vectors
    lat=[[1.0, 0.0], [0.0, 1.0]]
    # define coordinates of orbitals
    orb=[[0.0, 0.0], [0.5, 0.5]]

    # make two dimensional tight-binding checkerboard model
    model = tb_model(2, 2, lat=lat, orb=orb)

    # set on-site energies
    model.set_onsite([-delta, delta], mode='set')

    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])
    model.set_hop(-t0, 0, 0, [1, 0], mode='set')
    model.set_hop(-t0, 0, 0, [0, 1], mode='set')
    model.set_hop(t0, 1, 1, [1, 0], mode='set')
    model.set_hop(t0, 1, 1, [0, 1], mode='set')

    model.set_hop(tprime, 1, 0, [1, 1], mode='set')
    model.set_hop(tprime*1j, 1, 0, [0, 1], mode='set')
    model.set_hop(-tprime, 1, 0, [0, 0], mode='set')
    model.set_hop(-tprime*1j, 1, 0, [1, 0], mode='set')

    return model

def Haldane(delta, t, t2):
    lat=[[1, 0],[0.5, np.sqrt(3)/2]]
    orb=[[1/3, 1/3],[2/3, 2/3]]

    model = tb_model(2, 2, lat, orb)

    model.set_onsite([-delta, delta], mode='reset')

    for lvec in ([0, 0], [-1, 0], [0, -1]):
        model.set_hop(t, 0, 1, lvec, mode='reset')
        model.set_hop(t, 0, 1, lvec, mode='reset')

    for lvec in ([1, 0], [-1, 1], [0, -1]):
        model.set_hop(t2*1j, 0, 0, lvec, mode='reset')
        model.set_hop(t2*-1j, 1, 1, lvec, mode='reset')

    return model