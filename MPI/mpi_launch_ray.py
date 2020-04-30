from mpi_constants import *
import numpy as np

class Ray_XZ:
    def __init__(self, n, uray, boxes, marked, present,
                 x, z, crosses_x, crosses_z, edep, wpe, dedendx, dedendz,
                 x_init, z_init, kx_init, kz_init):
        self.raynum = n
        self.rayx = 0
        self.rayz = 0
        self.finalt = 0

        self.kx_init = kx_init
        self.kz_init = kz_init

        self.my_x = np.zeros(nt, dtype=np.float32, order='F')
        self.my_x[0] = x_init

        self.my_z = np.zeros(nt, dtype=np.float32, order='F')
        self.my_z[0] = z_init

        self.wpe = wpe

        self.marking_x = np.zeros(nt, dtype=int, order='F')
        self.marking_z = np.zeros(nt, dtype=int, order='F')
        self.x_bounds = np.zeros(nt, dtype=np.float32, order='F')
        self.z_bounds = np.zeros(nt, dtype=np.float32, order='F')
        self.x_bounds_double = np.zeros(nt, dtype=np.float32, order='F')
        self.z_bounds_double = np.zeros(nt, dtype=np.float32, order='F')

        self.my_vx = np.zeros(nt, dtype=np.float32, order='F')
        self.my_vz = np.zeros(nt, dtype=np.float32, order='F')

        self.launch_ray(uray, boxes, marked, present, x, z, edep, crosses_x, crosses_z, dedendx, dedendz)

    def get_finalt(self):
        return self.finalt

    def get_rayx(self):
        return self.rayx

    def get_rayz(self):
        return self.rayz

    def launch_ray(self, uray, boxes, marked, present, x, z, edep, crosses_x, crosses_z, dedendx, dedendz):
        for xx in range(nx):
            if ((-0.5 - 1.0e-10) * dx) <= (self.my_x[0] - x[xx, 0]) <= ((0.5 + 1.0e-10) * dx):
                thisx_0 = xx
                break  # "breaks" out of the xx loop once the if statement condition is met.

        for zz in range(nz):
            if ((-0.5 - 1.0e-10) * dz) <= (self.my_z[0] - z[0, zz]) <= ((0.5 + 1.0e-10) * dz):
                thisz_0 = zz
                break  # "breaks" out of the zz loop once the if statement condition is met.

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

        for tt in range(1, nt):
            self.my_vz[tt] = self.my_vz[tt - 1] - (c ** 2) / (2.0 * ncrit) * dedendz[thisx_0, thisz_0] * dt
            self.my_vx[tt] = self.my_vx[tt - 1] - (c ** 2) / (2.0 * ncrit) * dedendx[thisx_0, thisz_0] * dt

            self.my_x[tt] = self.my_x[tt - 1] + self.my_vx[tt] * dt
            self.my_z[tt] = self.my_z[tt - 1] + self.my_vz[tt] * dt

            search_index_x = 1  # Use nx for original (whole mesh) search
            search_index_z = 1  # Use nz for original (whole mesh) search

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

            linez = [self.my_z[tt - 1], self.my_z[tt]]  # In each loop, we analyze the current point and the point before
            linex = [self.my_x[tt - 1], self.my_x[tt]]

            lastx = 10000  # An initialization, more details below
            lastz = 10000  # An initialization, more details below

            for xx in range(thisx_m, thisx_p):
                currx = x[xx, 0] - (dx / 2)
                if (self.my_x[tt] > currx >= self.my_x[tt - 1]) or (self.my_x[tt] < currx <= self.my_x[tt - 1]):
                    crossx = np.interp(currx, linex, linez)
                    if abs(crossx - lastz) > 1.0e-20:
                        crosses_x[self.raynum, numcrossing] = currx
                        crosses_z[self.raynum, numcrossing] = crossx

                        if (xmin - dx / 2) <= self.my_x[tt] <= (xmax + dx / 2):
                            boxes[self.raynum, numcrossing, :] = [thisx, thisz]

                        lastx = currx
                        numcrossing += 1
                        break

            for zz in range(thisz_m, thisz_p):
                currz = z[0, zz] - (dz / 2)
                if (self.my_z[tt] > currz >= self.my_z[tt - 1]) or (self.my_z[tt] < currz <= self.my_z[tt - 1]):
                    crossz = np.interp(currz, linez, linex)
                    if abs(crossz - lastx) > 1.0e-20:
                        crosses_x[self.raynum, numcrossing] = crossz
                        crosses_z[self.raynum, numcrossing] = currz

                        if (zmin - dz / 2) <= self.my_z[tt] <= (zmax + dz / 2):
                            boxes[self.raynum, numcrossing, :] = [thisx, thisz]

                        numcrossing += 1
                        break

            thisx_0 = thisx
            thisz_0 = thisz

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
                    if marked[thisx, thisz, ss] == 0:
                        marked[thisx, thisz, ss] = self.raynum
                        present[thisx, thisz] += 1.0
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
                    if marked[thisx, thisz, ss] == 0:
                        marked[thisx, thisz, ss] = self.raynum
                        present[thisx, thisz] += 1.0
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
                    if marked[thisx, thisz, ss] == 0:
                        marked[thisx, thisz, ss] = self.raynum
                        present[thisx, thisz] += 1.0
                        break

            uray[tt] = uray[tt - 1]
            increment = uray[tt]

            xp = (self.my_x[tt] - (x[thisx, thisz] + dx / 2.0)) / dx
            zp = (self.my_z[tt] - (z[thisx, thisz] + dz / 2.0)) / dz

            if xp >= 0 and zp >= 0:
                dl = zp
                dm = xp
                a1 = (1.0 - dl) * (1.0 - dm)  # blue : (x, z)
                a2 = (1.0 - dl) * dm  # green : (x+1, z)
                a3 = dl * (1.0 - dm)  # yellow : (x, z+1)
                a4 = dl * dm  # red : (x+1, z+1)

                edep[thisx + 1, thisz + 1] += a1 * increment  # blue
                edep[thisx + 1 + 1, thisz + 1] += a2 * increment  # green
                edep[thisx + 1, thisz + 1 + 1] += a3 * increment  # yellow
                edep[thisx + 1 + 1, thisz + 1 + 1] += a4 * increment  # red
            elif xp < 0 and zp >= 0:
                dl = zp
                dm = abs(xp)
                a1 = (1.0 - dl) * (1.0 - dm)  # blue : (x, z)
                a2 = (1.0 - dl) * dm  # green : (x-1, z)
                a3 = dl * (1.0 - dm)  # yellow : (x, z+1)
                a4 = dl * dm  # red : (x-1, z+1)

                edep[thisx + 1, thisz + 1] += a1 * increment  # blue
                edep[thisx + 1 - 1, thisz + 1] += a2 * increment  # green
                edep[thisx + 1, thisz + 1 + 1] += a3 * increment  # yellow
                edep[thisx + 1 - 1, thisz + 1 + 1] += a4 * increment  # red
            elif xp >= 0 and zp < 0:
                dl = abs(zp)
                dm = xp
                a1 = (1.0 - dl) * (1.0 - dm)  # blue : (x, z)
                a2 = (1.0 - dl) * dm  # green : (x+1, z)
                a3 = dl * (1.0 - dm)  # yellow : (x, z-1)
                a4 = dl * dm  # red : (x+1, z-1)

                edep[thisx + 1, thisz + 1] += a1 * increment  # blue
                edep[thisx + 1 + 1, thisz + 1] += a2 * increment  # green
                edep[thisx + 1, thisz + 1 - 1] += a3 * increment  # yellow
                edep[thisx + 1 + 1, thisz + 1 - 1] += a4 * increment  # red
            elif xp < 0 and zp < 0:
                dl = abs(zp)
                dm = abs(xp)
                a1 = (1.0 - dl) * (1.0 - dm)  # blue : (x, z)
                a2 = (1.0 - dl) * dm  # green : (x-1, z)
                a3 = dl * (1.0 - dm)  # yellow : (x, z-1)
                a4 = dl * dm  # red : (x-1, z-1)

                edep[thisx + 1, thisz + 1] += a1 * increment  # blue
                edep[thisx + 1 - 1, thisz + 1] += a2 * increment  # green
                edep[thisx + 1, thisz + 1 - 1] += a3 * increment  # yellow
                edep[thisx + 1 - 1, thisz + 1 - 1] += a4 * increment  # red
            else:
                print(f'xp is {xp}, zp is {zp}')
                # edep[thisx, thisz, self.beam] += (self.nuei[tt] * (eden[thisx, thisz] / ncrit) * uray[tt - 1] * dt)
                print('***** ERROR in interpolation of laser deposition grid!! *****')
                break


            # print(f'{tt} edep: {edep}')

            if self.my_x[tt] < (xmin - (dx / 2.0)) or self.my_x[tt] > (xmax + (dx / 2.0)):
                self.finalt = tt - 1
                self.rayx = self.my_x[:self.finalt]
                self.rayz = self.my_z[:self.finalt]
                # elapsed_time('cat02')
                break  # "breaks" out of the tt loop once the if condition is satisfied
            elif self.my_z[tt] < (zmin - (dz / 2.0)) or self.my_z[tt] > (zmax + (dz / 2.0)):
                self.finalt = tt - 1
                self.rayx = self.my_x[:self.finalt]
                self.rayz = self.my_z[:self.finalt]
                # elapsed_time('cat02')
                break  # "breaks" out of the tt loop once the if condition is satisfied
