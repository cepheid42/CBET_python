import matplotlib.pyplot as plt
from constants import *
from numpy import max as npmax

cmap = 'jet'
''' plot electron density profile normalized to ncrit '''
def plot_everything(z, x, eden, mysaved_x, mysaved_z, finalts, intensity_sum, variable1, a0_variable):
    plt.figure()

    plt.pcolormesh(z, x, eden / ncrit, cmap=cmap)
    plt.plot(z - (dz / 2), x - (dx / 2), 'k--')
    plt.plot(x - (dx / 2), z - (dz / 2), 'k--')

    plt.plot(z - (dz / 2), x + (dx / 2), 'k--')
    plt.plot(x + (dx / 2), z - (dz / 2), 'k--')

    plt.plot(z + (dz / 2), x - (dx / 2), 'k--')
    plt.plot(x - (dx / 2), z + (dz / 2), 'k--')

    plt.plot(z + (dz / 2), x + (dx / 2), 'k--')
    plt.plot(x + (dx / 2), z + (dz / 2), 'k--')

    plt.plot(z, x, 'k--')
    plt.plot(x, z, 'k--')

    plt.colorbar()

    plt.xlabel('Z (cm)')
    plt.ylabel('X (cm)')
    plt.title('n_e_/n_crit_')

    plt.show(block=False)

    '''Plot the cumulative energy deposited to the array edep, which shares the dimensions of x, z, eden, dedendz, etc.'''
    for b in range(nbeams):
        for n in range(nrays):
            plt.plot(mysaved_z[:finalts[n, b], n, b], mysaved_x[:finalts[n, b], n, b], 'm')

    plt.show(block=False)

    plt.figure()

    clo = 0.0
    chi = npmax(intensity_sum)
    plt.pcolormesh(z, x, intensity_sum, cmap=cmap, vmin=clo, vmax=chi)
    plt.colorbar()
    plt.xlabel('Z (cm)')
    plt.ylabel('X (cm)')
    plt.title('Overlapped intensity')
    plt.show(block=False)

    plt.figure()

    plt.pcolormesh(z, x, variable1, cmap=cmap, vmin=0.0, vmax=0.021)
    plt.colorbar()
    plt.xlabel('Z (cm)')
    plt.ylabel('X (cm)')
    plt.title('Total original field amplitude (a0)')
    plt.show(block=False)



    plt.figure()
    plt.pcolormesh(z, x, a0_variable, cmap=cmap, vmin=0.0, vmax=0.021)
    plt.colorbar()
    plt.xlabel('Z (cm)')
    plt.ylabel('X (cm)')
    plt.title('Total CBET new field amplitude (a0)')
    plt.show(block=False)

    plt.figure()
    plt.plot(x[:, 0], a0_variable[:, 1], ',-b')
    plt.plot(x[:, 0], a0_variable[:, nz - 2], ',-r')
    plt.plot(x[:, 0], a0_variable[:, nz // 2], ',-g')
    plt.xlabel('X (cm)')
    plt.ylabel('a0')
    plt.title('a0(x) at z_min, z_0, z_max')
    plt.grid(linestyle='--')
    plt.show(block=False)

    plt.figure()
    plt.plot(z[0, :], a0_variable[1, :], ',-b')
    plt.plot(z[0, :], a0_variable[nx - 2, :], ',-r')
    plt.plot(z[0, :], a0_variable[nx // 2, :], ',-g')
    plt.xlabel('Z (cm)')
    plt.ylabel('a0')
    plt.title('a0(z) at x_min, x_0, x_max')
    plt.grid(linestyle='--')
    plt.show(block=False)