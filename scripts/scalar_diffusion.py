import ball_wrapper as ball
import ball128
import numpy as np
from   scipy.linalg      import eig
from scipy.sparse        import linalg as spla
import scipy.sparse      as sparse
import scipy.special     as spec
import dedalus.public as de
from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.core.distributor import Distributor
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import timesteppers
import pickle

import logging
logger = logging.getLogger(__name__)

def lambda_function(ell):
    return 1

# Gives LHS matrices for boussinesq
def matrices(N,ell):

    def D(mu,i,deg):
        if mu == +1: return B.op('D+',N,i,ell+deg)
        if mu == -1: return B.op('D-',N,i,ell+deg)

    def E(i,deg): return B.op('E',N,i,ell+deg)

    def C(deg): return ball128.connection(N,ell+deg,alpha_BC,2)

    Z = B.op('0',N,0,ell)

    M = E(1, 0).dot(E( 0, 0))

    M = M.tocsr()

    L = -D(-1,1,+1).dot(D(+1, 0, 0))

    L = L.tocsr()

    row = B.op('r=1',N,0,ell)*lambda_function(ell)

    tau = C( 0)[:,-1]

    col = tau.reshape((len(tau),1))

    L = sparse.bmat([[   L, col],
                     [ row,   0]])

    M = sparse.bmat([[     M, 0*col],
                     [0*row,      0]])

    L = L.tocsr()
    M = M.tocsr()

    return M, L

class StateVector:

    def __init__(self,T):
        self.data = []
        for ell in range(ell_start,ell_end+1):
            taus = np.zeros(1)
            ell_local = ell-ell_start
            for m in range(m_start,m_end+1):
                m_local = m-m_start
                self.data.append(np.concatenate((T['c'][ell_local][:,m_local],taus)))

    def pack(self,T,BC=None):
        if BC is None:
            BC = np.zeros(m_size, ell_size)
        for ell in range(ell_start,ell_end+1):
            ell_local = ell-ell_start
            for m in range(m_start,m_end+1):
                m_local = m-m_start
                self.data[ell_local*m_size+m_local] = np.concatenate((T['c'][ell_local][:,m_local],
                                                                      [BC[m_local,ell_local]]))

    def unpack(self,T):
        T.layout = 'c'
        for ell in range(ell_start,ell_end+1):
            ell_local = ell-ell_start
            for m in range(m_start,m_end+1):
                m_local = m - m_start
                T['c'][ell_local][:,m_local] = self.data[ell_local*m_size+m_local][:-1] #remove tau


# Find MPI rank
comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# Resolution
L_max = 31
N_max = 31
R_max = 2

alpha_BC = 0

L_dealias = 3/2
N_dealias = 3/2

# Integration parameters
dt = 8e-5
t_end = 20

# Make domain
mesh=[1]
phi_basis = de.Fourier('phi',2*(L_max+1), interval=(0,2*np.pi),dealias=L_dealias)
theta_basis = de.Fourier('theta', L_max+1, interval=(0,np.pi),dealias=L_dealias)
r_basis = de.Fourier('r', N_max+1, interval=(0,1),dealias=N_dealias)
domain = de.Domain([phi_basis,theta_basis,r_basis], grid_dtype=np.float64, mesh=mesh)

domain.global_coeff_shape = np.array([L_max+1,L_max+1,N_max+1])
domain.distributor = Distributor(domain,comm,mesh)

mesh = domain.distributor.mesh
if len(mesh) == 0:
    phi_layout   = domain.distributor.layouts[3]
    th_m_layout  = domain.distributor.layouts[2]
    ell_r_layout = domain.distributor.layouts[1]
    r_ell_layout = domain.distributor.layouts[1]
elif len(mesh) == 1:
    phi_layout   = domain.distributor.layouts[4]
    th_m_layout  = domain.distributor.layouts[2]
    ell_r_layout = domain.distributor.layouts[1]
    r_ell_layout = domain.distributor.layouts[1]
elif len(mesh) == 2:
    phi_layout   = domain.distributor.layouts[5]
    th_m_layout  = domain.distributor.layouts[3]
    ell_r_layout = domain.distributor.layouts[2]
    r_ell_layout = domain.distributor.layouts[1]

m_start   = th_m_layout.slices(scales=1)[0].start
m_end     = th_m_layout.slices(scales=1)[0].stop-1
m_size = m_end - m_start + 1
ell_start = r_ell_layout.slices(scales=1)[1].start
ell_end   = r_ell_layout.slices(scales=1)[1].stop-1
ell_size = ell_end - ell_start + 1

# set up ball
N_theta = int((L_max+1)*L_dealias)
N_r     = int((N_max+1)*N_dealias)
B = ball.Ball(N_max,L_max,N_theta=N_theta,N_r=N_r,R_max=R_max,ell_min=ell_start,ell_max=ell_end,m_min=m_start,m_max=m_end,a=0.)
theta_global = B.grid(0)
r_global = B.grid(1)
z, R = r_global*np.cos(theta_global), r_global*np.sin(theta_global) # global

grid_slices = phi_layout.slices(domain.dealias)
phi = domain.grid(0,scales=domain.dealias)[grid_slices[0],:,:]
theta = B.grid(1,dimensions=3)[:,grid_slices[1],:] # local
r = B.grid(2,dimensions=3)[:,:,grid_slices[2]] # local

weight_theta = B.weight(1,dimensions=3)[:,grid_slices[1],:]
weight_r = B.weight(2,dimensions=3)[:,:,grid_slices[2]]

# RHS BC
def f(m,ell):
    if ell==0: return 0
    else: return 0*ell**(-4)

# BC array
local_rell_shape = r_ell_layout.local_shape(scales=domain.dealias)
BC_shape = np.array(local_rell_shape)[:-1]
BC = np.zeros(BC_shape,dtype=np.complex128)

for ell in range(ell_start,ell_end+1):
    ell_local = ell-ell_start
    for m in range(m_start,m_end+1):
        m_local = m-m_start
        BC[m_local,ell_local] = f(m,ell)

T  = ball.TensorField_3D(0,B,domain)
T_rhs = ball.TensorField_3D(0,B,domain)

# initial condition
T['g'] = 0.5*(1-r**2) + 0.1/8.*np.sqrt(35/np.pi)*r**3*(1-r**2)*(np.cos(3*phi)+np.sin(3*phi))*np.sin(theta)**3

# build state vector
state_vector = StateVector(T)
NL = StateVector(T)
timestepper = timesteppers.SBDF2(StateVector, T)

# build matrices
M,L,P,LU = [],[],[],[]
for ell in range(ell_start,ell_end+1):
    N = B.N_max - B.N_min(ell-B.R_max)
    M_ell,L_ell = matrices(N,ell)
    M.append(M_ell.astype(np.complex128))
    L.append(L_ell.astype(np.complex128))
    P.append(M_ell.astype(np.complex128))
    LU.append([None])

# calculate RHS terms from state vector
def nonlinear(state_vector, RHS, t):

    # get U in coefficient space
    state_vector.unpack(T)

    T_rhs.layout = 'g'
    T_rhs['g'] = 0*T['g']

    # transform (ell, r) -> (ell, N)
    for ell in range(ell_start, ell_end+1):
        ell_local = ell - ell_start

        N = N_max - B.N_min(ell-R_max)

        # multiply by conversion matrices (may be very important)
        T_rhs['c'][ell_local] = M[ell_local][:-1,:-1]@T_rhs['c'][ell_local]

    NL.pack(T_rhs,BC)

reducer = GlobalArrayReducer(domain.dist.comm_cart)

t = 0.

t_list = []
E_list = []


def backward_state(state_vector):

    state_vector.unpack(T)
    T_global   = comm.gather(T['g'], root=0)

    starts = comm.gather(phi_layout.start(scales=domain.dealias),root=0)
    counts = comm.gather(phi_layout.local_shape(scales=domain.dealias),root=0)

    if rank == 0:
        T_full   = np.zeros(phi_layout.global_shape(scales=domain.dealias))
        for i in range(size):
            spatial_slices = tuple(slice(s, s+c) for (s,c) in zip(starts[i], counts[i]))
            T_full[spatial_slices]   = T_global[i]
    else:
        T_full   = None

    return T_full


# timestepping loop
start_time = time.time()
iter = 0

while t < t_end:
    # Output
    if iter % 10 == 0:
        Tmax = np.max(T['g'])
        Tmax = reducer.reduce_scalar(Tmax, MPI.MAX)
        logger.info("iter: {:d}, dt={:e}, t/t_e={:e}, Tmax={:e}".format(iter, dt, t/t_end, Tmax))
        if rank == 0:
            t_list.append(t)
    if iter % 100 == 0:
        T_grid = backward_state(state_vector)
        if rank == 0:
            output_num = iter // 5
            file = open('checkpoint_L%i' %output_num, 'wb')
            for a in [T_grid, phi, theta, r]:
                pickle.dump(a,file)
            file.close()
    # Timestep
    nonlinear(state_vector,NL,t)
    timestepper.step(dt, state_vector, B, L, M, P, NL, LU)
    t += dt
    iter += 1

end_time = time.time()
if rank==0:
    print('simulation took: %f' %(end_time-start_time))
    t_list = np.array(t_list)
    E_list = np.array(E_list)
    np.savetxt('E.dat',np.array([t_list,E_list]))

