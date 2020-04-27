'''==========================================================================================
                        2D RAY LAUNCHER AND TRACKER FUNCTION
                        Professor A. B. Sefkow
The essence of the routine is to enforce this governing equation:
w^2 = wpe(x,z)^2 + c^2*(kx^2+kz^2),
i.e. kx^2+kz^2 = c^-2*(w^2-wpe(x,z)^2),
where vg = dw/dk = c^2*k/w = (c/w)*sqrt(omega^2-wpe(x,z)^2)
And we update the ray positions (x) and group velocities (vg) according to
d(x)/dt = vg, and
d(vg)/dt = -c^2/2*(gradient(eden)/ncrit)
============================================================================================'''
from seq_constants import *
import numpy as np

class Ray_XZ:
    def __init__(self, beam, n, uray, boxes, marked, present,
                 x, z, crosses_x, crosses_z, edep, eden, dedendx, dedendz,
                 x_init, z_init, kx_init, kz_init):
        self.beam = beam
        self.raynum = n
        self.rayx = 0
        self.rayz = 0
        self.finalt = 0

        self.kx_init = kx_init
        self.kz_init = kz_init
        '''(x0, z0) is the initial starting point for the ray
            (kx0, kz0) is the initial wavevector for the ray 
            In ray.i, these were myx(1), myz(1), kx_init, and kz_init'''

        '''Set the initial location / position of the ray to be launched.
            The [0] is used because it is the first (time) step, which we define(initial condition)'''
        self.my_x = np.zeros(nt, dtype=np.float32, order='F')
        self.my_x[0] = x_init

        self.my_z = np.zeros(nt, dtype=np.float32, order='F')
        self.my_z[0] = z_init

        self.wpe = np.sqrt(eden * 1e6 * e_c**2.0 / (m_e * e_0))
        # self.nuei = np.ones(nt, dtype=np.float32, order='F')

        self.marking_x = np.zeros(nt, dtype=int, order='F')
        self.marking_z = np.zeros(nt, dtype=int, order='F')
        self.x_bounds = np.zeros(nt, dtype=np.float32, order='F')
        self.z_bounds = np.zeros(nt, dtype=np.float32, order='F')
        self.x_bounds_double = np.zeros(nt, dtype=np.float32, order='F')
        self.z_bounds_double = np.zeros(nt, dtype=np.float32, order='F')

        self.my_vx = np.zeros(nt, dtype=np.float32, order='F')
        self.my_vz = np.zeros(nt, dtype=np.float32, order='F')

        elapsed_time('cat02')

        self.launch_ray(uray, boxes, marked, present, x, z, crosses_x, crosses_z, edep, eden, dedendx, dedendz)

    def get_finalt(self):
        return self.finalt

    def get_rayx(self):
        return self.rayx

    def get_rayz(self):
        return self.rayz

    def launch_ray(self, uray, boxes, marked, present, x, z, crosses_x, crosses_z, edep, eden, dedendx, dedendz):
        """The following code will find the logical indices of the spatial locations on the mesh which
           correspond to those initial positions."""

        for xx in range(nx):
            if ((-0.5 - 1.0e-10) * dx) <= (self.my_x[0] - x[xx, 0]) <= ((0.5 + 1.0e-10) * dx):
                thisx_0 = xx
                thisx_00 = xx
                break  # "breaks" out of the xx loop once the if statement condition is met.

        for zz in range(nz):
            if ((-0.5 - 1.0e-10) * dz) <= (self.my_z[0] - z[0, zz]) <= ((0.5 + 1.0e-10) * dz):
                thisz_0 = zz
                thisz_00 = zz
                break  # "breaks" out of the zz loop once the if statement condition is met.

        elapsed_time('cat03')

        '''Calculate the total k( = sqrt(kx ^ 2 + kz ^ 2)) from the dispersion relation, 
           taking into account the local plasma frequency of where the ray starts.'''
        k = np.sqrt((omega ** 2 - self.wpe[thisx_0, thisz_0] ** 2) / c ** 2)

        knorm = np.sqrt(self.kx_init ** 2 + self.kz_init ** 2)  # Length of k for the ray to be launched


        mykx1 = (self.kx_init / knorm) * k
        mykz1 = (self.kz_init / knorm) * k
        self.my_vx[0] = (c ** 2) * mykx1 / omega  # v_group, group velocity (dw/dk) from D(k,w).
        self.my_vz[0] = (c ** 2) * mykz1 / omega

        self.marking_x[0] = thisx_0
        self.marking_z[0] = thisz_0
        self.x_bounds[0] = self.my_x[0]
        self.z_bounds[0] = self.my_z[0]

        numcrossing = 1

        elapsed_time('cat02')

        '''Begin the loop in time, where dt is the time step and each ray will be advanced in x and vg'''
        for tt in range(1, nt):
            '''We update the group velocities using grad(n_e), 
            and then the positions by doing this sequence:
            v_x = v_x0 + -c ^ 2 / (2 * nc) * d[ne(x)] / dx * dt
            v_z = v_z0 + -c ^ 2 / (2 * nc) * d[ne(z)] / dz * dt
            x_x = x_x0 + v_x * dt
            x_z = x_z0 + v_z * dt'''

            self.my_vz[tt] = self.my_vz[tt - 1] - (c ** 2) / (2.0 * ncrit) * dedendz[thisx_0, thisz_0] * dt
            self.my_vx[tt] = self.my_vx[tt - 1] - (c ** 2) / (2.0 * ncrit) * dedendx[thisx_0, thisz_0] * dt

            self.my_x[tt] = self.my_x[tt - 1] + self.my_vx[tt] * dt
            self.my_z[tt] = self.my_z[tt - 1] + self.my_vz[tt] * dt

            elapsed_time('cat05')

            '''============== Update the index, and track the intersections and boxes ================'''
            search_index_x = 1  # Use nx for original (whole mesh) search
            search_index_z = 1  # Use nz for original (whole mesh) search

            # thisx = None
            # thisz = None

            thisx_m = max(0, thisx_0 - search_index_x)
            thisx_p = min(nx, thisx_0 + search_index_x)

            thisz_m = max(0, thisz_0 - search_index_z)
            thisz_p = min(nz, thisz_0 + search_index_z)

            for xx in range(thisx_m, thisx_p + 1):  # Determines current x index for the position
                if xx >= len(x[:, 0]):
                    xx = len(x[:, 0]) - 1
                if (dx * (0.5 + 1.0e-10)) >= (self.my_x[tt] - x[xx, 0]) >= (-1 * (0.5 + 1.0e-10) * dx):
                    thisx = xx
                    break


            for zz in range(thisz_m, thisz_p + 1):  # Determines current z index for the position
                if zz >= len(z[:, 0]):
                    zz = len(z[:, 0]) - 1
                if (dz * (0.5 + 1.0e-10)) >= (self.my_z[tt] - z[0, zz]) >= (-1 * (0.5 + 1.0e-10) * dz):
                    thisz = zz
                    break

            elapsed_time('cat04')

            '''The (xi, zi) logical indices for the position are now stored as (thisx, thisz).
               We also need to reset thisx_0 and thisz_0 to the new indices thisx and thisz.'''
            linez = [self.my_z[tt - 1], self.my_z[tt]]  # In each loop, we analyze the current point and the point before
            linex = [self.my_x[tt - 1], self.my_x[tt]]

            lastx = 10000  # An initialization, more details below
            lastz = 10000  # An initialization, more details below

            for xx in range(thisx_m, thisx_p):
                currx = x[xx, 0] - (dx / 2)
                if (self.my_x[tt] > currx >= self.my_x[tt - 1]) or (self.my_x[tt] < currx <= self.my_x[tt - 1]):
                    crossx = np.interp(currx, linex, linez)
                    if abs(crossx - lastz) > 1.0e-20:
                        # self.ints[self.beam, self.raynum, numcrossing] = uray[tt]
                        crosses_x[self.beam, self.raynum, numcrossing] = currx
                        crosses_z[self.beam, self.raynum, numcrossing] = crossx

                        if (xmin - dx / 2) <= self.my_x[tt] <= (xmax + dx / 2):
                            boxes[self.beam, self.raynum, numcrossing, :] = [thisx, thisz]

                        lastx = currx
                        numcrossing += 1
                        break

            for zz in range(thisz_m, thisz_p):
                currz = z[0, zz] - (dz / 2)
                if (self.my_z[tt] > currz >= self.my_z[tt - 1]) or (self.my_z[tt] < currz <= self.my_z[tt - 1]):
                    crossz = np.interp(currz, linez, linex)
                    if abs(crossz - lastx) > 1.0e-20:
                        # self.ints[self.beam, self.raynum, numcrossing] = uray[tt]
                        crosses_x[self.beam, self.raynum, numcrossing] = crossz
                        crosses_z[self.beam, self.raynum, numcrossing] = currz

                        if (zmin - dz / 2) <= self.my_z[tt] <= (zmax + dz / 2):
                            boxes[self.beam, self.raynum, numcrossing, :] = [thisx, thisz]

                        lastz = currz   # This keeps track of the last z-value added, so that we don't double-count the corners
                        numcrossing += 1
                        break

            thisx_0 = thisx
            thisz_0 = thisz

            elapsed_time('cat08')

            self.marking_x[tt] = thisx
            self.marking_z[tt] = thisz

            if self.marking_x[tt] != self.marking_x[tt - 1] and self.marking_z[tt] != self.marking_z[tt - 1]:
                if self.my_vz[tt] < 0.0:
                    ztarg = z[thisx, thisz] + (dz / 2.0)
                else:
                    ztarg = z[thisx, thisz] - (dz / 2.0)

                slope = (self.my_z[tt] - self.my_z[tt - 1]) / (self.my_x[tt] - self.my_x[tt - 1] + 1.0e-10)
                xtarg = self.my_x[tt - 1] + (ztarg - self.my_z[tt - 1]) / slope
                self.x_bounds[tt] = xtarg
                self.z_bounds[tt] = ztarg

                if self.my_vx[tt] >= 0.0:
                    xtarg = x[thisx, thisz] - (dx / 2.0)
                else:
                    xtarg = x[thisx, thisz] + (dx / 2.0)

                slope = (self.my_x[tt] - self.my_x[tt - 1]) / (self.my_z[tt] - self.my_z[tt - 1] + 1.0e-10)
                ztarg = self.my_z[tt - 1] + (xtarg - self.my_x[tt - 1]) / slope
                self.x_bounds_double[tt] = xtarg
                self.z_bounds_double[tt] = ztarg

                for ss in range(numstored):
                    if marked[thisx, thisz, ss, self.beam] == 0:
                        marked[thisx, thisz, ss, self.beam] = self.raynum
                        present[thisx, thisz, self.beam] += 1.0
                        break

            elif self.marking_z[tt] != self.marking_z[tt - 1]:
                if self.my_vz[tt] < 0.0:
                    ztarg = z[thisx, thisz] + (dz / 2.0)
                else:
                    ztarg = z[thisx, thisz] - (dz / 2.0)

                slope = (self.my_z[tt] - self.my_z[tt - 1]) / (self.my_x[tt] - self.my_x[tt - 1] + 1.0e-10)
                xtarg = self.my_x[tt - 1] + (ztarg - self.my_z[tt - 1]) / slope
                self.x_bounds[tt] = xtarg
                self.z_bounds[tt] = ztarg

                for ss in range(numstored):
                    if marked[thisx, thisz, ss, self.beam] == 0:
                        marked[thisx, thisz, ss, self.beam] = self.raynum
                        present[thisx, thisz, self.beam] += 1.0
                        break

            elif self.marking_x[tt] != self.marking_x[tt - 1]:
                if self.my_vx[tt] >= 0.0:
                    xtarg = x[thisx, thisz] - (dx / 2.0)
                else:
                    xtarg = x[thisx, thisz] + (dx / 2.0)

                slope = (self.my_x[tt] - self.my_x[tt - 1]) / (self.my_z[tt] - self.my_z[tt - 1] + 1.0e-10)
                ztarg = self.my_z[tt - 1] + (xtarg - self.my_x[tt - 1]) / slope
                self.x_bounds[tt] = xtarg
                self.z_bounds[tt] = ztarg

                for ss in range(numstored):
                    if marked[thisx, thisz, ss, self.beam] == 0:
                        marked[thisx, thisz, ss, self.beam] = self.raynum
                        present[thisx, thisz, self.beam] += 1.0
                        break

            elapsed_time('cat12')

            '''In order to calculate the deposited energy into the plasma, 
               we need to calculate the plasma resistivity (eta) and collision frequency (nu_e-i)'''
            # eta = 5.2e-5 * 10.0 / (etemp(thisx, thisz) ^ 1.5) # ohm * m
            # nuei(tt) = (1e6 * eden(thisx, thisz) * ec ^ 2.0 / me) * eta

            '''Now we can decrement the ray's energy density according to how much energy 
               was absorbed by the plasma.'''
            # uray(tt) = uray(tt - 1) - (nuei(tt) * (eden(thisx, thisz) / ncrit) * uray(tt - 1) * dt)
            # increment = (nuei(tt) * (eden(thisx, thisz) / ncrit) * uray(tt - 1) * dt)

            '''We use these next two lines instead, if we are just using uray as a bookkeeping device 
               (i.e., no absorption by the plasma and no loss of energy by the ray).'''
            uray[tt] = uray[tt - 1]
            increment = uray[tt]

            '''Rather than put all the energy into the cell in which the ray resides, which is the 
               so-called "nearest-neighbor" approach (which is very noise and less accurate), we will 
               use an area-based linear weighting scheme to deposit the energy to the four nearest 
               nodes of the ray's current location. In 3D, this would be 8 nearest.'''
            # Define xp and zp to be the ray's position relative to the nearest node.
            xp = (self.my_x[tt] - (x[thisx, thisz] + dx / 2.0)) / dx
            zp = (self.my_z[tt] - (z[thisx, thisz] + dz / 2.0)) / dz


            # with open('xp_zp_py.csv', mode='a') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(f'{tt}, {xp:.10}, {zp:.10}')

            '''Below, we interpolate the energy deposition to the grid using linear area weighting. 
               The edep array must be two larger in each direction (one for min, one for max) 
               to accomodate this routine, since it deposits energy in adjacent cells.'''

            if xp >= 0 and zp >= 0:
                dl = zp
                dm = xp
                a1 = (1.0 - dl) * (1.0 - dm)  # blue : (x, z)
                a2 = (1.0 - dl) * dm  # green : (x+1, z)
                a3 = dl * (1.0 - dm)  # yellow : (x, z+1)
                a4 = dl * dm  # red : (x+1, z+1)

                edep[thisx + 1, thisz + 1, self.beam] += a1 * increment # blue
                edep[thisx + 1 + 1, thisz + 1, self.beam] += a2 * increment  # green
                edep[thisx + 1, thisz + 1 + 1, self.beam] += a3 * increment  # yellow
                edep[thisx + 1 + 1, thisz + 1 + 1, self.beam] += a4 * increment  # red
            elif xp < 0 and zp >= 0:
                dl = zp
                dm = abs(xp)
                a1 = (1.0 - dl) * (1.0 - dm)  # blue : (x, z)
                a2 = (1.0 - dl) * dm  # green : (x-1, z)
                a3 = dl * (1.0 - dm)  # yellow : (x, z+1)
                a4 = dl * dm  # red : (x-1, z+1)

                edep[thisx + 1, thisz + 1, self.beam] += a1 * increment  # blue
                edep[thisx + 1 - 1, thisz + 1, self.beam] += a2 * increment  # green
                edep[thisx + 1, thisz + 1 + 1, self.beam] += a3 * increment  # yellow
                edep[thisx + 1 - 1, thisz + 1 + 1, self.beam] += a4 * increment  # red
            elif xp >= 0 and zp < 0:
                dl = abs(zp)
                dm = xp
                a1 = (1.0 - dl) * (1.0 - dm)  # blue : (x, z)
                a2 = (1.0 - dl) * dm  # green : (x+1, z)
                a3 = dl * (1.0 - dm)  # yellow : (x, z-1)
                a4 = dl * dm  # red : (x+1, z-1)

                edep[thisx + 1, thisz + 1, self.beam] += a1 * increment  # blue
                edep[thisx + 1 + 1, thisz + 1, self.beam] += a2 * increment  # green
                edep[thisx + 1, thisz + 1 - 1, self.beam] += a3 * increment  # yellow
                edep[thisx + 1 + 1, thisz + 1 - 1, self.beam] += a4 * increment  # red
            elif xp < 0 and zp < 0:
                dl = abs(zp)
                dm = abs(xp)
                a1 = (1.0 - dl) * (1.0 - dm)  # blue : (x, z)
                a2 = (1.0 - dl) * dm  # green : (x-1, z)
                a3 = dl * (1.0 - dm)  # yellow : (x, z-1)
                a4 = dl * dm  # red : (x-1, z-1)
                edep[thisx + 1, thisz + 1, self.beam] += a1 * increment  # blue
                edep[thisx + 1 - 1, thisz + 1, self.beam] += a2 * increment  # green
                edep[thisx + 1, thisz + 1 - 1, self.beam] += a3 * increment  # yellow
                edep[thisx + 1 - 1, thisz + 1 - 1, self.beam] += a4 * increment  # red
            else:
                print(f'xp is {xp}, zp is {zp}')
                # edep[thisx, thisz, self.beam] += (self.nuei[tt] * (eden[thisx, thisz] / ncrit) * uray[tt - 1] * dt)
                print('***** ERROR in interpolation of laser deposition grid!! *****')
                break

            elapsed_time('cat06')

            # This will cause the code to stop following the ray once it escapes the extent of the plasma
            if self.my_x[tt] < (xmin - (dx / 2.0)) or self.my_x[tt] > (xmax + (dx / 2.0)):
                self.finalt = tt - 1
                self.rayx = self.my_x[:self.finalt]
                self.rayz = self.my_z[:self.finalt]
                elapsed_time('cat02')
                break  # "breaks" out of the tt loop once the if condition is satisfied
            elif self.my_z[tt] < (zmin - (dz / 2.0)) or self.my_z[tt] > (zmax + (dz / 2.0)):
                self.finalt = tt - 1
                self.rayx = self.my_x[:self.finalt]
                self.rayz = self.my_z[:self.finalt]
                elapsed_time('cat02')
                break  # "breaks" out of the tt loop once the if condition is satisfied