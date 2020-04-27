from numpy import pi, linspace, exp, sqrt, float32
from time import monotonic

# Dictionary for tracking timing information
start_time = monotonic()
timers = {f'cat{n + 1:02}': 0.0 for n in range(12)}
timers['start'] = start_time
timers['elapsed'] = start_time
timers['total'] = 0.0


def elapsed_time(cat):
    now = monotonic()
    timers[cat] += now - timers['elapsed']
    timers['elapsed'] = now


''' Define shape of ray space '''
nx = 201
xmin = -5.0e-4
xmax = 5.0e-4
dx = (xmax - xmin) / (nx - 1)

nz = 201
zmin = -5.0e-4
zmax = 5.0e-4
dz = (zmax - zmin) / (nz - 1)

nbeams = 2
ncrossings = nx * 3  # Maximum number of potential grid crossings by a ray

c = 29979245800.0  # speed of light in cm/s
e_0 = 8.85418782e-12  # permittivity of free space in m^-3 kg^-1 s^4 A^2
m_e = 9.10938356e-31  # electron mass in kg
e_c = 1.60217662e-19  # electron charge in C

''' 
Define the wavelength of the laser light, and hence its frequency and critical density
w^2 = w_p,e^2 + c^2*k^2 is the dispersion relation D(k,w) and w>w_p,e for real k 
'''
lamb = 1.053e-4 / 3.0  # wavelength of light, in cm. This is frequency-tripled "3w" or "blue" (UV) light
freq = c / lamb  # frequency of light, in Hz
omega = 2 * pi * freq  # frequency of light, in rad/s
ncrit = 1e-6 * (omega ** 2.0 * m_e * e_0 / e_c ** 2.0)

rays_per_zone = 5

beam_max_z = 3.0e-4
beam_min_z = -3.0e-4

nrays = int(rays_per_zone * (beam_max_z - beam_min_z) / dz) + 599

numstored = int(5 * rays_per_zone)  # max number of rays stored per zone

''' This is the spatial dependence used by Russ Follett's CBET test problem. '''
phase_x = linspace(beam_min_z, beam_max_z, nrays, dtype=float32)
sigma = 1.7e-4
pow_x = exp(-1 * ((phase_x / sigma) ** 2) ** (4.0 / 2.0))

'''
Set the time step to be a less than dx/c, the Courant condition.
Set the maximum number of time steps within the tracking function.
'''
offset = 0.5e-4
courant_mult = 0.2
dt = courant_mult * min(dx, dz) / c
nt = int(courant_mult**-1.0*max(nx, nz)*2.0)
intensity = 2.0e15  # intensity of the beam in W/cm^2

'''========= CBET INTERACTIONS, GAIN COEFFICIENTS, AND ITERATION OF SOLUTION  ====================='''
estat = 4.80320427e-10  # electron charge in statC
mach = -1.0 * sqrt(2)  # Mach number for max resonance
Z = 3.1  # ionization state
mi = 10230 * (1.0e3 * m_e)  # Mass of ion in g
mi_kg = 10230.0 * m_e  # Mass of ion in kg
Te = 2.0e3 * 11604.5052  # Temperature of electron in K
Te_eV = 2.0e3
Ti = 1.0e3 * 11604.5052  # Temperature of ion in K
Ti_eV = 1.0e3
iaw = 0.2  # ion-acoustic wave energy-damping rate (nu_ia/omega_s)!!
kb = 1.3806485279e-16  # Boltzmann constant in erg/K
kb2 = 1.3806485279e-23  # Boltzmann constant in J/K

constant1 = (estat ** 2) / (4 * (1.0e3 * m_e) * c * omega * kb * Te * (1 + 3 * Ti / (Z * Te)))

cs = 1e2 * sqrt(e_c * (Z * Te_eV + 3.0 * Ti_eV) / mi_kg)

elapsed_time('cat02')