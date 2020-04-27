import multiprocessing as mp
import numpy as np
from time import monotonic

from pool_constants import *
import pool_ray_launch as lr
# from plotter import plot_everything

start_time = monotonic()

num_procs = 2 # mp.cpu_count()

dedendz = np.zeros((nx, nz), dtype=np.float32, order='F')
dedendx = np.zeros((nx, nz), dtype=np.float32, order='F')

edep_x = np.zeros((nx + 2, nz + 2), dtype=np.float32, order='F')
edep_z = np.zeros((nx + 2, nz + 2), dtype=np.float32, order='F')

edep = np.zeros((nx + 2, nz + 2, nbeams), dtype=np.float32, order='F')

x = np.zeros((nx, nz), dtype=np.float32, order='F')
z = np.zeros((nx, nz), dtype=np.float32, order='F')

for zz in range(nz):
    x[:, zz] = np.linspace(xmin, xmax, nx, dtype=np.float32)

for xx in range(nx):
    z[xx, :] = np.linspace(zmin, zmax, nz, dtype=np.float32)


eden = np.zeros((nx, nz), dtype=np.float32, order='F')
machnum = np.zeros((nx, nz), dtype=np.float32, order='F')

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

uray = np.ones(nt, dtype=np.float32, order='F')

finalts = np.zeros((nrays, nbeams), dtype=np.int32, order='F')

mysaved_x = np.zeros((nt, nrays, nbeams), dtype=np.float32, order='F')
mysaved_z = np.zeros((nt, nrays, nbeams), dtype=np.float32, order='F')


marked = np.zeros((nx, nz, numstored, nbeams), dtype=np.int32, order='F')
present = np.zeros((nx, nz, nbeams), order='F')  # This array simply tallies the number of rays present in a zone from each beam

crosses_x = np.zeros((nbeams, nrays, ncrossings), dtype=np.float32, order='F')
crosses_z = np.zeros((nbeams, nrays, ncrossings), dtype=np.float32, order='F')
boxes = np.zeros((nbeams, nrays, ncrossings, 2), dtype=np.int32, order='F')

x0 = np.zeros(nrays, dtype=np.float32, order='F')
z0 = np.zeros(nrays, dtype=np.float32, order='F')

x0[:nrays] = xmin - (dt / courant_mult * c * 0.5)
z0[:nrays] = np.linspace(beam_min_z, beam_max_z, nrays, dtype=np.float32) + offset - (dz / 2) - (dt / courant_mult * c * 0.5)


kx0 = np.zeros(nrays, dtype=np.float32, order='F')
kz0 = np.zeros(nrays, dtype=np.float32, order='F')

kx0[:nrays] = np.float32(1.0)
kz0[:nrays] = np.float32(-0.1)

uray_mult = intensity * courant_mult * rays_per_zone**-1.0
wpe = np.sqrt(eden * 1e6 * e_c ** 2.0 / (m_e * e_0))

def lunch_ray(n, uray, uray_n, boxes, marked, present, x, z,
              crosses_x, crosses_z, edep, wpe, dedendx, dedendz,
              x_init, z_init, kx_init, kz_init):
    # parameters is all the values needed for each ray, hopefully
    # it gets unpacked in the ray class
    # returns finalt, rayx, rayz
    uray[0] = uray_n
    dummy = lr.Ray_XZ(n, uray, boxes, marked, present, x, z,
                      crosses_x, crosses_z, edep, wpe, dedendx, dedendz,
                      x_init, z_init, kx_init, kz_init)
    return dummy.get_values()


def first_calc(n, n_crossing, boxes_c, marked_c, present_c, intersects, dkx_b, dkz_b, dkmag_b, W1_c, W2_c, i_b1_b,
               eden_b):
    W1_new_b = []
    W2_new_b = []
    bb = 0

    for cc1 in range(n_crossing):
        if boxes_c[bb, n, cc1, 0] == 0 or boxes_c[bb, n, cc1, 1] == 0:
            break
        ix = boxes_c[bb, n, cc1, 0]
        iz = boxes_c[bb, n, cc1, 1]
        if intersects[ix, iz] != 0:
            nonzeros1 = marked_c[ix, iz, :, 0].nonzero()
            numrays1 = np.count_nonzero(marked_c[ix, iz, :, 0])

            nonzeros2 = marked_c[ix, iz, :, 1].nonzero()
            numrays2 = np.count_nonzero(marked_c[ix, iz, :, 1])

            marker1 = marked_c[ix, iz, nonzeros1, 0].flatten()
            marker2 = marked_c[ix, iz, nonzeros2, 1].flatten()

            rr2 = marker2
            cc2 = marker2

            for rrr in range(numrays1):
                if marker1[rrr] == n:
                    ray1num = rrr
                    break

            for n2 in range(numrays2):
                for ccc in range(n_crossing):
                    ix2 = boxes_c[bb + 1, rr2[n2], ccc, 0]
                    iz2 = boxes_c[bb + 1, rr2[n2], ccc, 1]
                    if ix == ix2 and iz == iz2:
                        cc2[n2] = ccc
                        break

            n2limit = int(min(present_c[ix, iz, 0], numrays2))

            for n2 in range(n2limit):
                ne = eden_b[ix, iz]
                epsilon = 1.0 - ne / ncrit
                kmag = (omega / c) * np.sqrt(epsilon)  # magnitude of wavevector

                kx1 = kmag * (dkx_b[bb, n, cc1] / (dkmag_b[bb, n, cc1] + 1.0e-10))
                kx2 = kmag * (dkx_b[bb + 1, rr2[n2], cc2[n2]] / (dkmag_b[bb + 1, rr2[n2], cc2[n2]] + 1.0e-10))

                kz1 = kmag * (dkz_b[bb, n, cc1] / (dkmag_b[bb, n, cc1] + 1.0e-10))
                kz2 = kmag * (dkz_b[bb + 1, rr2[n2], cc2[n2]] / (dkmag_b[bb + 1, rr2[n2], cc2[n2]] + 1.0e-10))

                kiaw = np.sqrt(
                    (kx2 - kx1) ** 2 + (kz2 - kz1) ** 2)  # magnitude of the difference between the two vectors
                ws = kiaw * cs  # acoustic frequency, cs is a constant
                omega1 = omega
                omega2 = omega  # laser frequency difference. zero to start

                eta = ((omega2 - omega1) - (kx2 - kx1) * u_flow[ix, iz]) / (ws + 1.0e-10)

                efield1 = np.sqrt(8.0 * np.pi * 1.0e7 * i_b1_b[ix, iz] / c)  # initial electric field of ray
                # efield2 = np.sqrt(8.0 * np.pi * 1.0e7 * i_b2[ix, iz] / c)  # initial electric field of ray

                P = (iaw ** 2 * eta) / ((eta ** 2 - 1.0) ** 2 + iaw ** 2 * eta ** 2)  # from Russ's paper
                gain2 = constant1 * efield1 ** 2 * (ne / ncrit) * (1 / iaw) * P  # L^-1 from Russ's paper

                # new energy of crossing (PROBE) ray (beam 2)
                if dkmag_b[bb + 1, rr2[n2], cc2[n2]] >= 1.0 * dx:
                    w2_new = W2_c[ix, iz] * np.exp(
                        -1 * W1_c[ix, iz] * dkmag_b[bb + 1, rr2[n2], cc2[n2]] * gain2 / np.sqrt(epsilon))
                    W2_new_b.append((w2_new, ix, iz))

                    w1_new = W1_c[ix, iz] * np.exp(1 * W2_c[ix, iz] * dkmag_b[bb, n, cc1] * gain2 / np.sqrt(epsilon))
                    W1_new_b.append((w1_new, ix, iz))

    return W1_new_b, W2_new_b


def second_calc(n, n_crossings, boxes_c, marked_c, intersects, W1_new_c, W2_new_c, W1_init_c, W2_init_c,
                i_b1_c, i_b2_c, x_c, z_c):
    bb = 0
    i_b1_new_c = np.zeros(i_b1_c.shape, order='F')
    i_b2_new_c = np.zeros(i_b2_c.shape, order='F')

    for cc1 in range(n_crossings):
        if boxes_c[bb, n, cc1, 0] == 0 or boxes_c[bb, n, cc1, 1] == 0:
            break
        ix = boxes_c[bb, n, cc1, 0]
        iz = boxes_c[bb, n, cc1, 1]

        if intersects[ix, iz] != 0:
            nonzeros1 = marked_c[ix, iz, :, 0].nonzero()
            numrays1 = np.count_nonzero(marked_c[ix, iz, :, 0])

            nonzeros2 = marked_c[ix, iz, :, 1].nonzero()
            numrays2 = np.count_nonzero(marked_c[ix, iz, :, 1])

            marker1 = marked_c[ix, iz, nonzeros1, 0].flatten()
            marker2 = marked_c[ix, iz, nonzeros2, 1].flatten()

            rr2 = marker2
            cc2 = marker2

            for rrr in range(numrays1):
                if marker1[rrr] == n:
                    ray1num = rrr
                    break

            for n2 in range(numrays2):
                for ccc in range(n_crossings):
                    ix2 = boxes_c[bb + 1, rr2[n2], ccc, 0]
                    iz2 = boxes_c[bb + 1, rr2[n2], ccc, 1]
                    if ix == ix2 and iz == iz2:
                        cc2[n2] = ccc
                        break

            fractional_change_1 = -1.0 * (1.0 - (W1_new_c[ix, iz] / W1_init_c[ix, iz])) * i_b1_c[ix, iz]
            fractional_change_2 = -1.0 * (1.0 - (W2_new_c[ix, iz] / W2_init_c[ix, iz])) * i_b2_c[ix, iz]

            i_b1_new_c[ix, iz] += fractional_change_1
            i_b2_new_c[ix, iz] += fractional_change_2

            x_prev_1 = x_c[ix, iz]
            z_prev_1 = z_c[ix, iz]

            x_prev_2 = x_c[ix, iz]
            z_prev_2 = z_c[ix, iz]

            # Now we need to find and increment/decrement the fractional_change for the rest of the beam 1 ray
            for ccc in range(cc1 + 1, n_crossings):
                ix_next_1 = boxes_c[0, n, ccc, 0]
                iz_next_1 = boxes_c[0, n, ccc, 1]

                x_curr_1 = x_c[ix_next_1, iz_next_1]
                z_curr_1 = z_c[ix_next_1, iz_next_1]

                if ix_next_1 == 0 or iz_next_1 == 0:
                    break
                else:
                    # Avoid double deposition if the (x,z) location doesn't change with incremented crossing number
                    if x_curr_1 != x_prev_1 or z_curr_1 != z_prev_1:
                        i_b1_new_c[ix_next_1, iz_next_1] += fractional_change_1 * (
                                    present_b[ix, iz, 0] / present_b[ix_next_1, iz_next_1, 0])

                    x_prev_1 = x_curr_1
                    z_prev_1 = z_curr_1

            n2 = min(ray1num, numrays2)

            for ccc in range(cc2[n2] + 1, n_crossings):
                ix_next_2 = boxes_c[1, rr2[n2], ccc, 0]
                iz_next_2 = boxes_c[1, rr2[n2], ccc, 1]

                x_curr_2 = x_c[ix_next_2, iz_next_2]
                z_curr_2 = z_c[ix_next_2, iz_next_2]

                if ix_next_2 == 0 or iz_next_2 == 0:
                    break
                else:
                    if x_curr_2 != x_prev_2 or z_curr_2 != z_prev_2:
                        i_b2_new_c[ix_next_2, iz_next_2] += fractional_change_2 * (
                                    present_b[ix, iz, 0] / present_b[ix_next_2, iz_next_2, 1])

                    x_prev_2 = x_curr_2
                    z_prev_2 = z_curr_2
    return i_b1_new_c, i_b2_new_c


# Beam pool, stores results in list
with mp.Pool(processes=num_procs) as pool:
    # Create initial uray weights
    uray_n0 = [uray_mult * np.interp(z0[n], phase_x + offset, pow_x) for n in range(nrays)]
    # Create iterable parameter list for passing to each function in pool
    params0 = [(n, uray, uray_n0[n], boxes[0, :, :, :], marked[:, :, :, 0], present[:, :, 0], x, z,
                crosses_x[0, :, :], crosses_z[0, :, :], edep[:, :, 0], wpe, dedendx, dedendz,
                x0[n], z0[n], kx0[n], kz0[n]) for n in range(nrays)]

    results0 = pool.starmap(lunch_ray, params0)

    # Extract results and assign them to correct arrays
    for n, val in enumerate(results0):
        finalt, rayx, rayz, edep_b, boxes_b, marked_b, present_b, crossesx_b, crossesz_b = val
        finalts[n, 0] = finalt
        mysaved_x[:finalt, n, 0] = rayx
        mysaved_z[:finalt, n, 0] = rayz
        edep[:, :, 0] += edep_b
        boxes[0, :, :, :] = boxes_b
        marked[:, :, :, 0] = marked_b
        present[:, :, 0] = present_b
        crosses_x[0, :, :] = crossesx_b
        crosses_z[0, :, :] = crossesz_b

    # clean up
    uray_n0 = None
    params0 = None
    results0 = None

    # Reset for next beam
    z0[:nrays] = zmin - (dt / courant_mult * c * 0.5)
    x0[:nrays] = np.linspace(beam_min_z, beam_max_z, nrays) - (dx / 2) - (dt / courant_mult * c * 0.5)
    kx0[:nrays] = 0.0
    kz0[:nrays] = 1.0

    # Create initial uray weights
    uray_n1 = [uray_mult * np.interp(z0[n], phase_x + offset, pow_x) for n in range(nrays)]
    # Create iterable parameter list for passing to each function in pool
    params1 = [(n, uray, uray_n1[n], boxes[1, :, :, :], marked[:, :, :, 1], present[:, :, 1], x, z,
                crosses_x[1, :, :], crosses_z[1, :, :], edep[:, :, 1], wpe, dedendx, dedendz,
                x0[n], z0[n], kx0[n], kz0[n]) for n in range(nrays)]

    results1 = pool.starmap(lunch_ray, params1)

    # Extract results and assign them to correct arrays
    for n, val in enumerate(results1):
        finalt, rayx, rayz, edep_b, boxes_b, marked_b, present_b, crossesx_b, crossesz_b = val
        finalts[n, 1] = finalt
        mysaved_x[:finalt, n, 1] = rayx
        mysaved_z[:finalt, n, 1] = rayz
        edep[:, :, 1] += edep_b
        boxes[1, :, :, :] = boxes_b
        marked[:, :, :, 1] = marked_b
        present[:, :, 1] = present_b
        crosses_x[1, :, :] = crossesx_b
        crosses_z[1, :, :] = crossesz_b

    uray_n1 = None
    params1 = None
    results1 = None

    i_b1 = np.copy(edep[:nx, :nz, 0], order='F')
    i_b2 = np.copy(edep[:nx, :nz, 1], order='F')

    intersections = np.zeros((nx, nz), dtype=np.float32, order='F')

    for xx in range(1, nx):  # loops start from 1, the first zone
        for zz in range(1, nz):
            for ss in range(numstored):
                if marked[xx, zz, ss, 0] == 0:
                    break
                else:
                    iray1 = marked[xx, zz, ss, 0]
                    for sss in range(numstored):
                        if marked[xx,zz, sss, 1] == 0:
                            break
                        else:
                            intersections[xx, zz] += 1.0

    u_flow = machnum * cs

    dkx = crosses_x[:, :, 1:] - crosses_x[:, :, :-1]
    dkz = crosses_z[:, :, 1:] - crosses_z[:, :, :-1]
    dkmag = np.sqrt(dkx ** 2 + dkz ** 2)

    W1 = np.sqrt(1 - eden / ncrit) / rays_per_zone
    W2 = np.sqrt(1 - eden / ncrit) / rays_per_zone

    W1_init = np.copy(W1, order='F')
    W1_new = np.copy(W1_init, order='F')

    W2_init = np.copy(W2, order='F')
    W2_new = np.copy(W2_init, order='F')

    i_b1_new = np.copy(i_b1, order='F')
    i_b2_new = np.copy(i_b2, order='F')

    params2 = [(n, ncrossings, boxes, marked, present, intersections, dkx, dkz, dkmag, W1, W2, i_b1, eden) for n in range(nrays)]

    results2 = pool.starmap(first_calc, params2)

    for n, ray in enumerate(results2):
        for i in ray[0]:  # W1_new_b
            val, ix, iz = i
            W1_new[ix, iz] = val

        for j in ray[1]:
            val, ix, iz = j
            W2_new[ix, iz] = val

    params2 = None
    results2 = None

    params3 = [(n, ncrossings, boxes, marked, intersections, W1_new, W2_new, W1_init, W2_init, i_b1, i_b2, x, z) for n in range(nrays)]

    results3 = pool.starmap(second_calc, params3)

    for n, ray in enumerate(results3):
        i_b1_new += ray[0]
        i_b2_new += ray[1]

    params3 = None
    results3 = None



intensity_sum = np.sum(edep[:nx, :nz, :], axis=2)
variable1 = 8.53e-10 * np.sqrt(i_b1 + i_b2 + 1.0e-10) * (1.053 / 3.0)
i_b1_new[i_b1_new < 1.0e-10] = 1.0e-10
i_b2_new[i_b2_new < 1.0e-10] = 1.0e-10
a0_variable = 8.53e-10 * np.sqrt(i_b1_new + i_b2_new + 1.0e-10) * (1.053 / 3.0)

# plot_everything(z, x, eden, mysaved_x, mysaved_z, finalts, intensity_sum, variable1, a0_variable)

'''==================== TIMER REPORTS ============================================================='''
print("FINISHED!    Reporting ray timings now...")
print('___________________________________________________________________')
print(f'Total time: {monotonic() - start_time}')