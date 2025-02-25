from pythtb import *
from numpy import sqrt
import numpy as np
import sys
sys.path.append("../wanpy")
from wpythtb import Model

# used for testing purposes

def checkerboard(t0, tprime, delta):
    # define lattice vectors
    lat=[[1.0, 0.0], [0.0, 1.0]]
    # define coordinates of orbitals
    orb=[[0.0, 0.0], [0.5, 0.5]]

    # make two dimensional tight-binding checkerboard model
    model = Model(2, 2, lat=lat, orb=orb)

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
    lat = [[1, 0], [0.5, sqrt(3)/2]]
    orb = [[1/3, 1/3], [2/3, 2/3]]

    model = Model(2, 2, lat, orb)

    model.set_onsite([-delta, delta], mode='reset')

    for lvec in ([0, 0], [-1, 0], [0, -1]):
        model.set_hop(t, 0, 1, lvec, mode='reset')
        model.set_hop(t, 0, 1, lvec, mode='reset')

    for lvec in ([1, 0], [-1, 1], [0, -1]):
        model.set_hop(t2*1j, 0, 0, lvec, mode='reset')
        model.set_hop(t2*-1j, 1, 1, lvec, mode='reset')

    return model


def kagome(t1, t2):

    lat_vecs = [[1, 0], [1/2, np.sqrt(3)/2]]
    orb_vecs = [[0,0], [1/2, 0], [0, 1/2]]

    model = Model(2, 2, lat_vecs, orb_vecs)

    model.set_hop(t1+1j*t2, 0, 1, [0, 0])
    model.set_hop(t1+1j*t2, 2, 0, [0, 0])
    model.set_hop(t1+1j*t2, 0, 1, [-1, 0])
    model.set_hop(t1+1j*t2, 2, 0, [0, 1])
    model.set_hop(t1+1j*t2, 1, 2, [0, 0])
    model.set_hop(t1+1j*t2, 1, 2, [1, -1])

    return model


def kane_mele(onsite, t, soc, rashba):
  "Return a Kane-Mele model in the normal or topological phase."

  # define lattice vectors
  lat = [[1.0,0.0],[0.5,sqrt(3.0)/2.0]]
  # define coordinates of orbitals
  orb = [[1./3.,1./3.],[2./3.,2./3.]]

  # make two dimensional tight-binding Kane-Mele model
  ret_model = Model(2, 2, lat, orb, nspin=2)

  # set on-site energies
  ret_model.set_onsite([onsite, -onsite])

  # useful definitions
  sigma_x = np.array([0.,1.,0.,0])
  sigma_y = np.array([0.,0.,1.,0])
  sigma_z = np.array([0.,0.,0.,1])

  # set hoppings (one for each connected pair of orbitals)
  # (amplitude, i, j, [lattice vector to cell containing j])
  # spin-independent first-neighbor hoppings
  ret_model.set_hop(t, 0, 1, [ 0, 0])
  ret_model.set_hop(t, 0, 1, [ 0,-1])
  ret_model.set_hop(t, 0, 1, [-1, 0])

  # second-neighbour spin-orbit hoppings (s_z)
  ret_model.set_hop(-1.j*soc*sigma_z, 0, 0, [ 0, 1])
  ret_model.set_hop( 1.j*soc*sigma_z, 0, 0, [ 1, 0])
  ret_model.set_hop(-1.j*soc*sigma_z, 0, 0, [ 1,-1])
  ret_model.set_hop( 1.j*soc*sigma_z, 1, 1, [ 0, 1])
  ret_model.set_hop(-1.j*soc*sigma_z, 1, 1, [ 1, 0])
  ret_model.set_hop( 1.j*soc*sigma_z, 1, 1, [ 1,-1])

  # Rashba first-neighbor hoppings: (s_x)(dy)-(s_y)(d_x)
  r3h = np.sqrt(3.0)/2.0
  # bond unit vectors are (r3h,half) then (0,-1) then (-r3h,half)
  ret_model.set_hop(1.j*rashba*( 0.5*sigma_x-r3h*sigma_y), 0, 1, [ 0, 0], mode="add")
  ret_model.set_hop(1.j*rashba*(-1.0*sigma_x            ), 0, 1, [ 0,-1], mode="add")
  ret_model.set_hop(1.j*rashba*( 0.5*sigma_x+r3h*sigma_y), 0, 1, [-1, 0], mode="add")

  return ret_model


def fu_kane_mele(t, soc, m, beta):
    # set up Fu-Kane-Mele model
    lat = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    orb = [[0, 0, 0], [0.25, 0.25, 0.25]]
    model = Model(3, 3, lat, orb, nspin=2)

    h = m*np.sin(beta)*np.array([1,1,1])
    dt = m*np.cos(beta)

    h0 = [0] + list(h)
    h1 = [0] + list(-h)

    model.set_onsite(h0, 0)
    model.set_onsite(h1, 1)

    # spin-independent first-neighbor hops
    for lvec in ([-1, 0, 0], [0, -1, 0], [0, 0, -1]):
        model.set_hop(t, 0, 1, lvec)

    model.set_hop(3*t + dt, 0, 1, [0, 0, 0], mode="add")

    # spin-dependent second-neighbor hops
    lvec_list = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 1, 0], [0, -1, 1], [1, 0, -1])
    dir_list = ([0, 1, -1], [-1, 0, 1], [1, -1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1])
    for j in range(6):
        spin = np.array([0.]+dir_list[j])
        model.set_hop( 1j*soc*spin, 0, 0, lvec_list[j])
        model.set_hop(-1j*soc*spin, 1, 1, lvec_list[j])

    return model





