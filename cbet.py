from mpi4py import MPI

from constants import *
import launch_ray as lr
from plotter import plot_everything

import numpy as np
from time import monotonic

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    start_time = monotonic()

x = np.zeros((nx, nz), dtype=np.float32, order='F')
z = np.zeros((nx, nz), dtype=np.float32, order='F')
dedendz = np.zeros((nx, nz), dtype=np.float32, order='F')
dedendx = np.zeros((nx, nz), dtype=np.float32, order='F')
machnum = np.zeros((nx, nz), dtype=np.float32, order='F')
eden = np.zeros((nx, nz), dtype=np.float32, order='F')
finalts = np.zeros((nrays, nbeams), dtype=np.int32, order='F')
mysaved_x = np.zeros((nt, nrays, nbeams), dtype=np.float32, order='F')
mysaved_z = np.zeros((nt, nrays, nbeams), dtype=np.float32, order='F')
edep = np.zeros((nx + 2, nz + 2, nbeams), dtype=np.float32, order='F')
marked = np.zeros((nx, nz, numstored, nbeams), dtype=np.int32, order='F')
crosses_x = np.zeros((nbeams, nrays, ncrossings), dtype=np.float32, order='F')
crosses_z = np.zeros((nbeams, nrays, ncrossings), dtype=np.float32, order='F')
boxes = np.zeros((nbeams, nrays, ncrossings, 2), dtype=np.int32, order='F')
present = np.zeros((nx, nz, nbeams), dtype=np.int32, order='F')

loc_edep = np.zeros((nx + 2, nz + 2), dtype=np.float32, order='F')
loc_marked = np.zeros((nx, nz, numstored), dtype=np.int32, order='F')
loc_crosses_x = np.zeros((nrays, ncrossings), dtype=np.float32, order='F')
loc_crosses_z = np.zeros((nrays, ncrossings), dtype=np.float32, order='F')
loc_boxes = np.zeros((nrays, ncrossings, 2), dtype=np.int32, order='F')
loc_present = np.zeros((nx, nz), dtype=np.int32, order='F')
loc_finalts = np.zeros(nrays, dtype=np.int32, order='F')
loc_savedx = np.zeros((nt, nrays), dtype=np.float32, order='F')
loc_savedz = np.zeros((nt, nrays), dtype=np.float32, order='F')

if rank == 0:
    for zz in range(nz):
        x[:, zz] = np.linspace(xmin, xmax, nx, dtype=np.float32)

    for xx in range(nx):
        z[xx, :] = np.linspace(zmin, zmax, nz, dtype=np.float32)

    print('More initialization...')

    for xx in range(nx):
        for zz in range(nz):
            eden[xx, zz] = max(0.0, ((0.3 * ncrit - 0.1 * ncrit) / (xmax - xmin)) * (x[xx, zz] - xmin) + (0.1 * ncrit))
            machnum[xx, zz] = max(0.0, (((-0.4) - (-2.4)) / (xmax - xmin)) * (x[xx, zz] - xmin)) + (-2.4)

    for xx in range(nx - 1):
        for zz in range(nz - 1):
            dedendz[xx, zz] = (eden[xx, zz + 1] - eden[xx, zz]) / (z[xx, zz + 1] - z[xx, zz])
            dedendx[xx, zz] = (eden[xx + 1, zz] - eden[xx, zz]) / (x[xx + 1, zz] - x[xx, zz])

    dedendz[:, nz - 1] = dedendz[:, nz - 2]  # sets last column equal to second to last column
    dedendx[nx - 1, :] = dedendz[nx - 2, :]  # sets last row equal to second to last row

    print('Setting initial conditions for ray tracker')
    print('nrays per beam is ', nrays)

comm.Bcast(x, root=0)
comm.Bcast(z, root=0)
comm.Bcast(eden, root=0)
comm.Bcast(machnum, root=0)
comm.Bcast(dedendx, root=0)
comm.Bcast(dedendz, root=0)

uray = np.ones(nt, dtype=np.float32, order='F')

x0 = np.zeros(nrays, dtype=np.float32, order='F')
z0 = np.zeros(nrays, dtype=np.float32, order='F')

kx0 = np.zeros(nrays, dtype=np.float32, order='F')
kz0 = np.zeros(nrays, dtype=np.float32, order='F')

uray_mult = intensity * courant_mult * rays_per_zone**-1.0
wpe = np.sqrt(eden * 1e6 * e_c ** 2.0 / (m_e * e_0))

print("Tracking Rays...")

if rank == 0:
    x0[:] = xmin - (dt / courant_mult * c * 0.5)
    z0[:] = np.linspace(beam_min_z, beam_max_z, nrays, dtype=np.float32) + offset - (dz / 2) - (dt / courant_mult * c * 0.5)

    kx0[:nrays] = np.float32(1.0)
    kz0[:nrays] = np.float32(-0.1)

    print('BEAMNUM is ', rank + 1)
    for n in range(nrays):  # loop over rays
        uray[0] = uray_mult * np.interp(z0[n], phase_x + offset, pow_x)  # determines initial power weighting

        dummy = lr.Ray_XZ(n, uray, loc_boxes, loc_marked, loc_present,
                          x, z, loc_crosses_x, loc_crosses_z, loc_edep, wpe, dedendx, dedendz,
                          x0[n], z0[n], kx0[n], kz0[n])

        finalt = dummy.get_finalt()
        rayx = dummy.get_rayx()
        rayz = dummy.get_rayz()

        loc_finalts[n] = finalt
        loc_savedx[:finalt, n] = rayx
        loc_savedz[:finalt, n] = rayz

        if n % 20 == 0:
            print(f'     ...{int(100 * (1 - (n / nrays)))}% remaining...')

if rank == 1:
    x0[:] = np.linspace(beam_min_z, beam_max_z, nrays, dtype=np.float32) - (dx / 2) - (dt / courant_mult * c * 0.5)
    z0[:] = zmin - (dt / courant_mult * c * 0.5)

    kx0[:nrays] = np.float32(0.0)
    kz0[:nrays] = np.float32(1.0)

    print('BEAMNUM is ', rank + 1)
    for n in range(nrays):  # loop over rays
        uray[0] = uray_mult * np.interp(x0[n], phase_x, pow_x)  # determines initial power weighting

        dummy = lr.Ray_XZ(n, uray, loc_boxes, loc_marked, loc_present,
                          x, z, loc_crosses_x, loc_crosses_z, loc_edep, wpe, dedendx, dedendz,
                          x0[n], z0[n], kx0[n], kz0[n])

        finalt = dummy.get_finalt()
        rayx = dummy.get_rayx()
        rayz = dummy.get_rayz()

        loc_finalts[n] = finalt
        loc_savedx[:finalt, n] = rayx
        loc_savedz[:finalt, n] = rayz

        if n % 20 == 0:
            print(f'     ...{int(100 * (1 - (n / nrays)))}% remaining...')

if rank == 1:
    comm.Send(loc_crosses_x, dest=0, tag=15)
    comm.Send(loc_crosses_z, dest=0, tag=16)
    comm.Send(loc_edep, dest=0, tag=17)
    comm.Send(loc_marked, dest=0, tag=18)
    comm.Send(loc_boxes, dest=0, tag=19)
    comm.Send(loc_present, dest=0, tag=20)
    comm.Send(loc_finalts, dest=0, tag=21)
    comm.Send(loc_savedx, dest=0, tag=22)
    comm.Send(loc_savedz, dest=0, tag=23)

if rank == 0:
    temp_crossx = np.empty((nrays, ncrossings), dtype=np.float32, order='F')
    temp_crossz = np.empty((nrays, ncrossings), dtype=np.float32, order='F')
    temp_edep = np.empty((nx + 2, nz + 2), dtype=np.float32, order='F')
    temp_marked = np.empty((nx, nz, numstored), dtype=np.int32, order='F')
    temp_boxes = np.empty((nrays, ncrossings, 2), dtype=np.int32, order='F')
    temp_present = np.empty((nx, nz), dtype=np.int32, order='F')
    temp_finalts = np.empty(nrays, dtype=np.int32, order='F')
    temp_savedx = np.empty((nt, nrays), dtype=np.float32, order='F')
    temp_savedz = np.empty((nt, nrays), dtype=np.float32, order='F')

    comm.Recv(temp_crossx, source=1, tag=15)
    comm.Recv(temp_crossz, source=1, tag=16)
    comm.Recv(temp_edep, source=1, tag=17)
    comm.Recv(temp_marked, source=1, tag=18)
    comm.Recv(temp_boxes, source=1, tag=19)
    comm.Recv(temp_present, source=1, tag=20)
    comm.Recv(temp_finalts, source=1, tag=21)
    comm.Recv(temp_savedx, source=1, tag=22)
    comm.Recv(temp_savedz, source=1, tag=23)

    crosses_x[1, :, :] = temp_crossx
    crosses_z[1, :, :] = temp_crossz
    edep[:, :, 1] = temp_edep
    marked[:, :, :, 1] = temp_marked
    boxes[1, :, :, :] = temp_boxes
    present[:, :, 1] = temp_present
    finalts[:, 1] = temp_finalts
    mysaved_x[:, :, 1] = temp_savedx
    mysaved_z[:, :, 1] = temp_savedz

    crosses_x[0, :, :] = loc_crosses_x
    crosses_z[0, :, :] = loc_crosses_z
    edep[:, :, 0] = loc_edep
    marked[:, :, :, 0] = loc_marked
    boxes[0, :, :, :] = loc_boxes
    present[:, :, 0] = loc_present
    finalts[:, 0] = loc_finalts
    mysaved_x[:, :, 0] = loc_savedx
    mysaved_z[:, :, 0] = loc_savedz

comm.Bcast(edep, root=0)
comm.Bcast(crosses_x, root=0)
comm.Bcast(crosses_z, root=0)
comm.Bcast(boxes, root=0)
comm.Bcast(present, root=0)


i_b1 = np.copy(edep[:nx, :nz, 0], order='F')
i_b2 = np.copy(edep[:nx, :nz, 1], order='F')

print("Finding ray intersections with rays from opposing beams.")
intersections = np.zeros((nx, nz), dtype=np.float32, order='F')

if rank == 0:
    for xx in range(1, nx):  # loops start from 1, the first zone
        for zz in range(1, nz):
            for ss in range(numstored):
                if marked[xx, zz, ss, 0] == 0:
                    break
                else:
                    # iray1 = marked[xx, zz, ss, 0]
                    for sss in range(numstored):
                        if marked[xx,zz, sss, 1] == 0:
                            break
                        else:
                            intersections[xx, zz] += 1.0

comm.Bcast(intersections, root=0)

if rank == 0:
    print('Calculating CBET gains...')


dkx = crosses_x[:, :, 1:] - crosses_x[:, :, :-1]
dkz = crosses_z[:, :, 1:] - crosses_z[:, :, :-1]
dkmag = np.sqrt(dkx ** 2 + dkz ** 2)
u_flow = machnum * cs

W1 = np.sqrt(1 - eden / ncrit) / rays_per_zone
W2 = np.sqrt(1 - eden / ncrit) / rays_per_zone

W1_init = np.copy(W1, order='F')
W1_new = np.copy(W1_init, order='F')
W2_init = np.copy(W2, order='F')
W2_new = np.copy(W2_init, order='F')

W1_storage = np.zeros((nx, nz, numstored), dtype=np.float32, order='F')
W2_storage = np.zeros((nx, nz, numstored), dtype=np.float32, order='F')

for bb in range(nbeams - 1):
    for rr1 in range(nrays):
        for cc1 in range(ncrossings):
            if rank == 0:
                if boxes[bb, rr1, cc1, 0] == 0 or boxes[bb, rr1, cc1, 1] == 0:
                    break

                ix = boxes[bb, rr1, cc1, 0]
                iz = boxes[bb, rr1, cc1, 1]

                if intersections[ix, iz] != 0:
                    nonzeros1 = marked[ix, iz, :, 0].nonzero()
                    numrays1 = np.count_nonzero(marked[ix, iz, :, 0])

                    nonzeros2 = marked[ix, iz, :, 1].nonzero()
                    numrays2 = np.count_nonzero(marked[ix, iz, :, 1])

                    marker1 = marked[ix, iz, nonzeros1, 0].flatten()
                    marker2 = marked[ix, iz, nonzeros2, 1].flatten()

                    rr2 = marker2
                    cc2 = marker2

                    for rrr in range(numrays1):
                        if marker1[rrr] == rr1:
                            ray1num = rrr
                            break

                    for n2 in range(numrays2):
                        for ccc in range(ncrossings):
                            ix2 = boxes[bb + 1, rr2[n2], ccc, 0]
                            iz2 = boxes[bb + 1, rr2[n2], ccc, 1]
                            if ix == ix2 and iz == iz2:
                                cc2[n2] = ccc
                                break

                    rr2_shape = rr2.shape
                    cc2_shape = cc2.shape
                    for rank in range(1, size):
                        comm.send(rr2_shape, dest=rank, tag=11)
                        comm.send(cc2_shape, dest=rank, tag=12)
                        # comm.send(ray1num, dest=rank, tag=13)

                    n2limit = int(min(present[ix, iz, 0], numrays2))
            else:
                n2limit = None
                # ray1num = None
                ix = None
                iz = None
                rr2_shape = comm.recv(source=0, tag=11)
                cc2_shape = comm.recv(source=0, tag=12)

                rr2 = np.empty(rr2_shape)
                cc2 = np.empty(cc2_shape)

            comm.Bcast(rr2, root=0)
            comm.Bcast(cc2, root=0)

            comm.bcast(n2limit, root=0)
            comm.bcast(ix, root=0)
            comm.bcast(iz, root=0)
            # comm.bcast(ray1num, root=0)

            for n2 in range(rank, n2limit, size):
                ne = eden[ix, iz]
                epsilon = 1.0 - ne / ncrit
                kmag = (omega / c) * np.sqrt(epsilon)  # magnitude of wavevector

                kx1 = kmag * (dkx[bb, rr1, cc1] / (dkmag[bb, rr1, cc1] + 1.0e-10))
                kx2 = kmag * (dkx[bb + 1, rr2[n2], cc2[n2]] / (dkmag[bb + 1, rr2[n2], cc2[n2]] + 1.0e-10))

                kz1 = kmag * (dkz[bb, rr1, cc1] / (dkmag[bb, rr1, cc1] + 1.0e-10))
                kz2 = kmag * (dkz[bb + 1, rr2[n2], cc2[n2]] / (dkmag[bb + 1, rr2[n2], cc2[n2]] + 1.0e-10))

                kiaw = np.sqrt((kx2 - kx1) ** 2 + (kz2 - kz1) ** 2)  # magnitude of the difference between the two vectors
                ws = kiaw * cs  # acoustic frequency, cs is a constant
                omega1 = omega
                omega2 = omega  # laser frequency difference. zero to start

                eta = ((omega2 - omega1) - (kx2 - kx1) * u_flow[ix, iz]) / (ws + 1.0e-10)

                efield1 = np.sqrt(8.0 * np.pi * 1.0e7 * i_b1[ix, iz] / c)  # initial electric field of ray
                efield2 = np.sqrt(8.0 * np.pi * 1.0e7 * i_b2[ix, iz] / c)  # initial electric field of ray

                P = (iaw ** 2 * eta) / ((eta ** 2 - 1.0) ** 2 + iaw ** 2 * eta ** 2)  # from Russ's paper
                gain1 = constant1 * efield2 ** 2 * (ne / ncrit) * (1 / iaw) * P  # L^-1 from Russ's paper
                gain2 = constant1 * efield1 ** 2 * (ne / ncrit) * (1 / iaw) * P  # L^-1 from Russ's paper

                if dkmag[bb + 1, rr2[n2], cc2[n2]] >= 1.0 * dx:
                    W2_new_ix_iz = W2[ix, iz] * np.exp(-1 * W1[ix, iz] * dkmag[bb + 1, rr2[n2], cc2[n2]] * gain2 / np.sqrt(epsilon))
                    W1_new_ix_iz = W1[ix, iz] * np.exp(1 * W2[ix, iz] * dkmag[bb, rr1, cc1] * gain2 / np.sqrt(epsilon))

                    if rank != 0:
                        comm.send(n2, dest=0, tag=10)
                        comm.send(W2_new_ix_iz, dest=0, tag=12)
                        comm.send(W1_new_ix_iz, dest=0, tag=13)
                    else:
                        for r in range(1, size):
                            index = comm.recv(source=r, tag=10)
                            temp1 = comm.recv(source=r, tag=12)
                            W2_storage[ix, iz, index] = temp1

                            temp2 = comm.recv(source=r, tag=13)
                            W1_storage[ix, iz, ray1num] = temp2


        if rr1 % 20 == 0:
            print(f'     ...{int(100 * (1 - (rr1 / nrays)))}%  remaining...')

if rank == 0:
    print("Updating intensities due to CBET gains...")

    i_b1_new = np.copy(i_b1, order='F')
    i_b2_new = np.copy(i_b2, order='F')

    for bb in range(nbeams - 1):
        for rr1 in range(nrays):
            for cc1 in range(ncrossings):
                if boxes[bb, rr1, cc1, 0] == 0 or boxes[bb, rr1, cc1, 1] == 0:
                    break
                ix = boxes[bb, rr1, cc1, 0]
                iz = boxes[bb, rr1, cc1, 1]

                if intersections[ix, iz] != 0:
                    nonzeros1 = marked[ix, iz, :, 0].nonzero()
                    numrays1 = np.count_nonzero(marked[ix, iz, :, 0])

                    nonzeros2 = marked[ix, iz, :, 1].nonzero()
                    numrays2 = np.count_nonzero(marked[ix, iz, :, 1])

                    marker1 = marked[ix, iz, nonzeros1, 0].flatten()
                    marker2 = marked[ix, iz, nonzeros2, 1].flatten()

                    rr2 = marker2
                    cc2 = marker2

                    for rrr in range(numrays1):
                        if marker1[rrr] == rr1:
                            ray1num = rrr
                            break

                    for n2 in range(numrays2):
                        for ccc in range(ncrossings):
                            ix2 = boxes[bb + 1, rr2[n2], ccc, 0]
                            iz2 = boxes[bb + 1, rr2[n2], ccc, 1]
                            if ix == ix2 and iz == iz2:
                                cc2[n2] = ccc
                                break

                    fractional_change_1 = -1.0 * (1.0 - (W1_new[ix, iz] / W1_init[ix, iz])) * i_b1[ix, iz]
                    fractional_change_2 = -1.0 * (1.0 - (W2_new[ix, iz] / W2_init[ix, iz])) * i_b2[ix, iz]

                    i_b1_new[ix, iz] += fractional_change_1
                    i_b2_new[ix, iz] += fractional_change_2

                    x_prev_1 = x[ix, iz]
                    z_prev_1 = z[ix, iz]

                    x_prev_2 = x[ix, iz]
                    z_prev_2 = z[ix, iz]

                    # Now we need to find and increment/decrement the fractional_change for the rest of the beam 1 ray
                    for ccc in range(cc1 + 1, ncrossings):
                        ix_next_1 = boxes[0, rr1, ccc, 0]
                        iz_next_1 = boxes[0, rr1, ccc, 1]

                        x_curr_1 = x[ix_next_1, iz_next_1]
                        z_curr_1 = z[ix_next_1, iz_next_1]

                        if ix_next_1 == 0 or iz_next_1 == 0:
                            break
                        else:
                            # Avoid double deposition if the (x,z) location doesn't change with incremented crossing number
                            if x_curr_1 != x_prev_1 or z_curr_1 != z_prev_1:
                                i_b1_new[ix_next_1, iz_next_1] += fractional_change_1 * (present[ix, iz, 0] / present[ix_next_1, iz_next_1, 0])

                            x_prev_1 = x_curr_1
                            z_prev_1 = z_curr_1

                    n2 = min(ray1num, numrays2)

                    for ccc in range(cc2[n2] + 1, ncrossings):
                        ix_next_2 = boxes[1, rr2[n2], ccc, 0]
                        iz_next_2 = boxes[1, rr2[n2], ccc, 1]

                        x_curr_2 = x[ix_next_2, iz_next_2]
                        z_curr_2 = z[ix_next_2, iz_next_2]

                        if ix_next_2 == 0 or iz_next_2 == 0:
                            break
                        else:
                            if x_curr_2 != x_prev_2 or z_curr_2 != z_prev_2:
                                i_b2_new[ix_next_2, iz_next_2] += fractional_change_2 * (present[ix, iz, 0] / present[ix_next_2, iz_next_2, 1])

                            x_prev_2 = x_curr_2
                            z_prev_2 = z_curr_2
            if rr1 % 20 == 0:
                print(f'     ...{int(100 * (1 - (rr1 / nrays)))}%  remaining...')


    intensity_sum = np.sum(edep[:nx, :nz, :], axis=2)
    variable1 = 8.53e-10 * np.sqrt(i_b1 + i_b2 + 1.0e-10) * (1.053 / 3.0)
    i_b1_new[i_b1_new < 1.0e-10] = 1.0e-10
    i_b2_new[i_b2_new < 1.0e-10] = 1.0e-10
    a0_variable = 8.53e-10 * np.sqrt(i_b1_new + i_b2_new + 1.0e-10) * (1.053 / 3.0)

    plot_everything(z, x, eden, mysaved_x, mysaved_z, finalts, intensity_sum, variable1, a0_variable)

    '''==================== TIMER REPORTS ============================================================='''
    print("FINISHED!    Reporting ray timings now...")
    print('___________________________________________________________________')
    print(f'Total time: {monotonic() - start_time}')
