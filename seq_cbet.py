from seq_constants import *
import seq_ray_launch as lr
# from plotter import plot_everything

import numpy as np

# Define 2D arrays that will store data for electron density, derivatives of e_den, and x/z
dedendz = np.zeros((nx, nz), dtype=np.float32, order='F')    # Backwards, because it is transposed later
dedendx = np.zeros((nx, nz), dtype=np.float32, order='F')

edep_x = np.zeros((nx + 2, nz + 2), dtype=np.float32, order='F')
edep_z = np.zeros((nx + 2, nz + 2), dtype=np.float32, order='F')

edep = np.zeros((nx + 2, nz + 2, nbeams), dtype=np.float32, order='F')

# Define 2D arrays of x and z spatial coordinates
x = np.zeros((nx, nz), dtype=np.float32, order='F')
z = np.zeros((nx, nz), dtype=np.float32, order='F')

for zz in range(nz):
    x[:, zz] = np.linspace(xmin, xmax, nx, dtype=np.float32)

for xx in range(nx):
    z[xx, :] = np.linspace(zmin, zmax, nz, dtype=np.float32)


# print('More initialization...')

''' Calculate the electron density using a function of x and z, as desired. '''
eden = np.zeros((nx, nz), dtype=np.float32, order='F')
machnum = np.zeros((nx, nz), dtype=np.float32, order='F')

for xx in range(nx):
    for zz in range(nz):
        eden[xx, zz] = max(0.0, ((0.3 * ncrit - 0.1 * ncrit) / (xmax - xmin)) * (x[xx, zz] - xmin) + (0.1 * ncrit))
        machnum[xx, zz] = max(0.0, (((-0.4) - (-2.4)) / (xmax - xmin)) * (x[xx, zz] - xmin)) + (-2.4)


''' Calculate the gradients of electron density w.r.t. x and z '''
for xx in range(nx - 1):
    for zz in range(nz - 1):
        dedendz[xx, zz] = (eden[xx, zz + 1] - eden[xx, zz]) / (z[xx, zz + 1] - z[xx, zz])
        dedendx[xx, zz] = (eden[xx + 1, zz] - eden[xx, zz]) / (x[xx + 1, zz] - x[xx, zz])

dedendz[:, nz - 1] = dedendz[:, nz - 2]  # sets last column equal to second to last column
dedendx[nx - 1, :] = dedendz[nx - 2, :]  # sets last row equal to second to last row

elapsed_time('cat02')

'''========== CODE TO TRACK RAY PROPAGATION IN THE EIKONAL APPROXIMATION ====================='''

# print('Setting initial conditions for ray tracker')
#
# print('nrays per beam is ', nrays)

'''
Define the x and z, k_x and k_z, and v_x and v_z arrays to store them for the ray.
They are the positions, the wavevectors, and the group velocities
'''
uray = np.ones(nt, dtype=np.float32, order='F')

finalts = np.zeros((nrays, nbeams), dtype=np.int32, order='F')

mysaved_x = np.zeros((nt, nrays, nbeams), dtype=np.float32, order='F')
mysaved_z = np.zeros((nt, nrays, nbeams), dtype=np.float32, order='F')

'''
The bookkeeping array is named "marked". The first two sets of elements denote the zone number,
labeled from 0 to (nx-1) and 0 to (nz-1).
The third set of elements is the list of rays that passed through the zone.
'''

marked = np.zeros((nx, nz, numstored, nbeams), dtype=np.int32, order='F')
present = np.zeros((nx, nz, nbeams), order='F')  # This array simply tallies the number of rays present in a zone from each beam

crosses_x = np.zeros((nbeams, nrays, ncrossings), dtype=np.float32, order='F')
crosses_z = np.zeros((nbeams, nrays, ncrossings), dtype=np.float32, order='F')
boxes = np.zeros((nbeams, nrays, ncrossings, 2), dtype=np.int32, order='F')

''' Set the initial location/position of the ray to be launched. '''
x0 = np.zeros(nrays, dtype=np.float32, order='F')
z0 = np.zeros(nrays, dtype=np.float32, order='F')

x0[:nrays] = xmin - (dt / courant_mult * c * 0.5)
z0[:nrays] = np.linspace(beam_min_z, beam_max_z, nrays, dtype=np.float32) + offset - (dz / 2) - (dt / courant_mult * c * 0.5)

'''
Set the initial unnormalized k vectors, which give the initial direction of the launched ray.
For example, kx = kz = 1 would give a 45 degree angle in the +x / +z direction.
For example, kx = 0 (or kz = 0) would give pure kz (or kx) propagation.
'''
kx0 = np.zeros(nrays, dtype=np.float32, order='F')
kz0 = np.zeros(nrays, dtype=np.float32, order='F')

kx0[:nrays] = np.float32(1.0)
kz0[:nrays] = np.float32(-0.1)

uray_mult = intensity * courant_mult * rays_per_zone**-1.0

# Initialization timer
elapsed_time('cat02')

''' Begin the loop over rays, to launch multiple rays in different directions from different initial locations. '''

# print("Tracking Rays...")

beam = 0
# print('BEAMNUM is ', beam + 1)
for n in range(nrays):  # loop over rays
    uray[0] = uray_mult * np.interp(z0[n], phase_x + offset, pow_x)  # determines initial power weighting
    dummy = lr.Ray_XZ(beam, n, uray, boxes, marked, present,
                      x, z, crosses_x, crosses_z, edep, eden, dedendx, dedendz,
                      x0[n], z0[n], kx0[n], kz0[n])

    finalt = dummy.get_finalt()
    rayx = dummy.get_rayx()
    rayz = dummy.get_rayz()

    finalts[n, beam] = finalt
    mysaved_x[:finalt, n, beam] = rayx
    mysaved_z[:finalt, n, beam] = rayz

    # if n % 20 == 0:
    #     print(f'     ...{int(100 * (1 - (n / nrays)))}% remaining...')

    elapsed_time('cat07')

# The following simply re-defines the x0,z0,kx0,kz0 for additional beams to launch.
''' These may not be needed if I pass copies to beam launcher'''
z0[:nrays] = zmin - (dt / courant_mult * c * 0.5)
x0[:nrays] = np.linspace(beam_min_z, beam_max_z, nrays) - (dx / 2) - (dt / courant_mult * c * 0.5)

kx0[:nrays] = 0.0
kz0[:nrays] = 1.0

beam = 1
# print('BEAMNUM is ', beam + 1)
for n in range(nrays):  # loop over rays
    uray[0] = uray_mult * np.interp(x0[n], phase_x, pow_x)  # determines initial power weighting
    dummy = lr.Ray_XZ(beam, n, uray, boxes, marked, present,
                      x, z, crosses_x, crosses_z, edep, eden, dedendx, dedendz,
                      x0[n], z0[n], kx0[n], kz0[n])

    finalt = dummy.get_finalt()
    rayx = dummy.get_rayx()
    rayz = dummy.get_rayz()

    finalts[n, beam] = finalt
    mysaved_x[:finalt, n, beam] = rayx
    mysaved_z[:finalt, n, beam] = rayz

    # if n % 20 == 0:
    #     print(f'     ...{int(100 * (1 - (n / nrays)))}% remaining...')

    elapsed_time('cat07')

# Beam launching timer
elapsed_time('cat07')

i_b1 = np.copy(edep[:nx, :nz, 0], order='F')
i_b2 = np.copy(edep[:nx, :nz, 1], order='F')

'''========== FINDING AND SAVING ALL POINTS OF INTERSECTION BETWEEN THE BEAMS  =====================
Loop through the spatial grid (2:nx,2:nz) to find where the rays from opposite beams intersect.
The dimensions of the array are: marked( nx, nz, raynum, beamnum)
// for raynum:  ss = 1; ss <= numstored; ++ss
// for beamnum: bb = 1; bb <= nbeams; ++bb
There will always be a beam #1, so start with beam 1 and check for rays present in the zone
from other beams. Starting from beam #1, need to check for rays from beam #2, then #3, etc.
After beam #1, need to start from beam #2 and check for rays from beam #3, then #4, etc.
NOTE: Starting from beam #2, do NOT need to re-check for rays from beam #1.
'''

# print("Finding ray intersections with rays from opposing beams.")
intersections = np.zeros((nx, nz), dtype=np.float32, order='F')

for xx in range(1, nx):  # loops start from 1, the first zone
    for zz in range(1, nz):
        # irays = 1
        # for b in range(nbeams):
            # irays *= np.count_nonzero(marked[xx, zz, :, b])  # finds number of nonzero entries in marked for each beam. multiplies them together for total intersections
        # intersections[xx, zz] = irays

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

# Intersection timer
elapsed_time('cat09')

# print('Calculating CBET gains...')

u_flow = machnum * cs

# Find wavevectors, normalize, then multiply by magnitude.
# crossesx and crossesz have dimensions (nbeams, nrays, ncrossings)

''' Second index slice are to -1, because its not inclusive (so it leaves out last column) '''
dkx = crosses_x[:, :, 1:] - crosses_x[:, :, :-1]
dkz = crosses_z[:, :, 1:] - crosses_z[:, :, :-1]
dkmag = np.sqrt(dkx ** 2 + dkz ** 2)

W1 = np.sqrt(1 - eden / ncrit) / rays_per_zone
W2 = np.sqrt(1 - eden / ncrit) / rays_per_zone

W1_init = np.copy(W1, order='F')
W1_new = np.copy(W1_init, order='F')
W2_init = np.copy(W2, order='F')
W2_new = np.copy(W2_init, order='F')

elapsed_time('cat02')
# The PROBE beam gains (loses) energy when gain2 < 0  (>0)
# The PUMP beam loses (gains) energy when gain2 < 0  (>0)

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

                n2limit = int(min(present[ix, iz, 0], numrays2))

                for n2 in range(n2limit):
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
                    # efield2 = np.sqrt(8.0 * np.pi * 1.0e7 * i_b2[ix, iz] / c)  # initial electric field of ray

                    P = (iaw ** 2 * eta) / ((eta ** 2 - 1.0) ** 2 + iaw ** 2 * eta ** 2)  # from Russ's paper
                    gain2 = constant1 * efield1 ** 2 * (ne / ncrit) * (1 / iaw) * P  # L^-1 from Russ's paper

                    # new energy of crossing (PROBE) ray (beam 2)
                    if dkmag[bb + 1, rr2[n2], cc2[n2]] >= 1.0 * dx:
                        W2_new[ix, iz] = W2[ix, iz] * np.exp(-1 * W1[ix, iz] * dkmag[bb + 1, rr2[n2], cc2[n2]] * gain2 / np.sqrt(epsilon))

                        W1_new[ix, iz] = W1[ix, iz] * np.exp(1 * W2[ix, iz] * dkmag[bb, rr1, cc1] * gain2 / np.sqrt(epsilon))

        # if rr1 % 20 == 0:
        #     print(f'     ...{int(100 * (1 - (rr1 / nrays)))}%  remaining...')

elapsed_time('cat11')

# print("Updating intensities due to CBET gains...")

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
        # if rr1 % 20 == 0:
        #     print(f'     ...{int(100 * (1 - (rr1 / nrays)))}%  remaining...')

elapsed_time('cat11')

intensity_sum = np.sum(edep[:nx, :nz, :], axis=2)
variable1 = 8.53e-10 * np.sqrt(i_b1 + i_b2 + 1.0e-10) * (1.053 / 3.0)
i_b1_new[i_b1_new < 1.0e-10] = 1.0e-10
i_b2_new[i_b2_new < 1.0e-10] = 1.0e-10
a0_variable = 8.53e-10 * np.sqrt(i_b1_new + i_b2_new + 1.0e-10) * (1.053 / 3.0)

# plot_everything(z, x, eden, mysaved_x, mysaved_z, finalts, intensity_sum, variable1, a0_variable)

'''==================== TIMER REPORTS ============================================================='''
# print("FINISHED!    Reporting ray timings now...")
# print('___________________________________________________________________')

elapsed_time('cat10')

ray_loop_sum = timers['cat03'] + timers['cat04'] + timers['cat05'] + timers['cat06'] + timers['cat08']
other_times = 0.0
for n in range(12):
    other_times += timers[f'cat{n+1:02}']

timers['total'] = monotonic() - timers['start']

others = timers['total'] - other_times

print(f'Data Import:                                {timers["cat01"]:15.8f}\n\
Initialization:                             {timers["cat02"]:15.8f}\n\
Initial ray index search:                   {timers["cat03"]:15.8f}\n\
Index search in ray timeloop:               {timers["cat04"]:15.8f}\n\
Ray push:                                   {timers["cat05"]:15.8f}\n\
Mapping ray trajectories to grid cat08:     {timers["cat08"]:15.8f}\n\
Mapping ray trajectories to grid cat12:     {timers["cat12"]:15.8f}\n\
Interpolation for deposition:               {timers["cat06"]:15.8f}\n\
Ray loops sum:                              {ray_loop_sum:15.8f}\n\
Finding intersections:                      {timers["cat09"]:15.8f}\n\
Plotting ray information:                   {timers["cat07"]:15.8f}\n\
CBET gain calculations:                     {timers["cat11"]:15.8f}\n\
Others...:                                  {others:15.8f}\n\
TOTAL:                                      {timers["total"]:15.8f}')
print('-------------------------------------------------------------------')