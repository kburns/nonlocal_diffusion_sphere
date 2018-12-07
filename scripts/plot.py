
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import pathlib
import pickle
from dedalus.extras import plot_tools
from mpi4py import MPI


def top_plot(phi, theta, data, pcm=None, cmap=None):
    r = np.sin(theta)
    # Plot
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    r, phi = plot_tools.quad_mesh(r.ravel(), phi.ravel())
    r[:, 0] = 0
    pcm = ax.pcolormesh(phi, r, data, cmap=cmap)
    ax.set_aspect(1)
    plt.colorbar(pcm)


def equator_plot(phi, r, data, **kw):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    r, phi = plot_tools.quad_mesh(r.ravel(), phi.ravel())
    r[:, 0] = 0
    pcm = ax.pcolormesh(phi, r, data, **kw)
    ax.set_aspect(1)
    plt.colorbar(pcm)




cmap = 'RdBu_r'

file_paths = sorted(pathlib.Path('.').glob('checkpoints/checkpoint_*'))
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

for file_path in file_paths[rank::size]:
    print(rank, file_path)
    # Load data
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    r = data['r']
    theta = data['theta']
    phi = data['phi']
    t = data['time']
    T = data['T']
    # Equator plot
    n_theta = theta.size
    if n_theta % 2 == 0:
        T_eq = (T[:, (n_theta-1)//2, :] + T[:, (n_theta+1)//2, :]) / 2
    else:
        T_eq = T[:, (n_theta-1)//2, :]
    equator_plot(phi, r, T_eq, cmap=cmap, clim=(-20, 20))
    plt.savefig('frames/equator_%06i.png' %data['iteration'])
    # Top plot
    # n_r = r.size
    # print(theta[theta < np.pi/2])
    # print(theta.shape, T.shape)
    # T_mid = T[theta < np.pi/2][:, :, n_r//2]
    # top_plot(phi, theta[theta < np.pi/2], T_mid, cmap=cmap)
    # plt.savefig('frames/top_%6i.png' %data['iteration'])

