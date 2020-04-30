/*==========================================================================================
			2D RAY-TRACING PROGRAM FOR CROSS-BEAM ENERGY TRANSFER
			Professor A. B. Sefkow
============================================================================================*/

/* Arrays needed for tracking timing information */
elapsed= elapsed0= total= array(double, 3);
cat01= cat02= cat03 = cat04= cat05= cat06= cat07= cat08= cat09= cat10= array(double, 3);
cat11= cat12= cat13 = cat14= cat15= cat16= cat17= cat18= cat19= cat20= array(double, 3);

timer, elapsed0; // Defines the start time for the timer
elapsed= elapsed0;

func init_win{;winkill,0;winkill,1;winkill,2;winkill,3;
window,0,dpi=180;window,1,dpi=180;window,2,dpi=180;window,3,dpi=60;};

windows=0
if ( windows == 1){; init_win;};

/*===========================================================================================*/

#include "palettes/pal.i"
#include "palettes/pal2.i"
#include "palettes/colorbar.i"
#include "launch_ray_XZ.i"

hcpon;
//=========== INITIALIZATIONS ================================================================//

/* Define the 2D cartesian extent of the grid in cm (1e-4 is 1 micron). */
nx = 201;
xmin = -5.0e-4;
xmax = 5.0e-4;
dx = (xmax-xmin)/(nx-1);
nz = 201;
zmin = -5.0e-4;
zmax = 5.0e-4;
dz = (zmax-zmin)/(nz-1);

nbeams = 2;

/* Define 2D arrays that will store data for electron density, derivatives of e_den, and x/z */
eden = array(0.0,nx,nz);
dedendz = array(0.0,nx,nz);
dedendx = array(0.0,nx,nz);
x = array(0.0,nx,nz);
z = array(0.0,nx,nz);
edep_x = array(0.0,nx+2,nz+2);
edep_z = array(0.0,nx+2,nz+2);

edep = array(0.0,nx+2,nz+2, nbeams);    // needs to be one zone bigger at xmin/xmax and zmin/zmax
                                // to accommodate ray deposition interpolation
machnum = array(0.0,nx,nz);

/* Define 2D arrays of x and z spatial coordinates */
for (zz=1;zz<=nz;++zz){;
	x(,zz) = span(xmin,xmax,nx);
};
for (xx=1;xx<=nx;++xx){;
	z(xx,) = span(zmin,zmax,nz);
};

/* print, "More initialization..." */

/* Define some constants to be used later */
c=29979245800.0; 		// speed of light in cm/s
e0=8.85418782e-12;	// permittivity of free space in m^-3 kg^-1 s^4 A^2
me=9.10938356e-31;	// electron mass in kg
ec=1.60217662e-19;	// electron charge in C

/* Define the wavelength of the laser light, and hence its frequency and critical density */
/* w^2 = w_p,e^2 + c^2*k^2 is the dispersion relation D(k,w) and w>w_p,e for real k */

lambda = 1.053e-4/3.0;	// wavelength of light, in cm. This is frequency-tripled "3w" or "blue" (UV) light
freq = c/lambda;		// frequency of light, in Hz
omega = 2*pi*freq;	// frequency of light, in rad/s
ncrit = 1e-6*(omega^2.0*me*e0/ec^2.0);	// the critical density occurs when omega = omega_p,e

/* Define 2D arrays that will store data for the derivatives of e_den */
/* dedendz = array(0.0,nx,nz);
dedendx = array(0.0,nx,nz); */

/* Calculate the electron density using a function of x and z, as desired. */
for (xx=1;xx<=nx;++xx){;
        for (zz=1;zz<=nz;++zz){;
            eden(xx,zz) = max(0.0,((0.3*ncrit-0.1*ncrit)/(xmax-xmin))*(x(xx,zz)-xmin)+(0.1*ncrit));
            machnum(xx,zz) = max(0.0,(((-0.4)-(-2.4))/(xmax-xmin))*(x(xx,zz)-xmin))+(-2.4);
        };
};

/* Calculate the gradients of electron density w.r.t. x and z */
for (xx=1;xx<=nx-1;++xx){
	for (zz=1;zz<=nz-1;++zz){
		dedendz(xx,zz) = (eden(xx,zz+1)-eden(xx,zz))/(z(xx,zz+1)-z(xx,zz))
		dedendx(xx,zz) = (eden(xx+1,zz)-eden(xx,zz))/(x(xx+1,zz)-x(xx,zz))
	};
};
dedendz(,nz) = dedendz(,nz-1);
dedendx(nx,) = dedendx(nx-1,);
/* The above lines define the derivatives in the last zones as being equal to the
next-to-last zones. */

/* Calculate the electron plasma frequency and normalized electron density (to ncrit) */
wpe = sqrt(eden*1e6*ec^2.0/(me*e0));

// Plots the electron density profile normalized to ncrit
window,1;
fma;
clo=0.1
chi=0.3
plf,eden/ncrit,x,z,cmin=clo,cmax=chi

plm,x-(dx/2),z-(dz/2),type="dash";
plm,x-(dx/2),z+(dz/2),type="dash";
plm,x+(dx/2),z-(dz/2),type="dash";
plm,x+(dx/2),z+(dz/2),type="dash";
plm,x,z,width=2;
xytitles,"Z (cm)","X (cm)";
pltitle,"n_e_/n_crit_";
colorbar,clo,chi;
p9;

timer, elapsed, cat02;		// initialization timer category

//========== CODE TO TRACK RAY PROPAGATION IN THE EIKONAL APPROXIMATION =====================//

/* print, "Setting initial conditions for ray tracker..." */

/*  Set the number of rays to be launched to represent a beam. */
rays_per_zone = 5 ;

beam_max_z = 3.0e-4;
beam_min_z = -3.0e-4;
nrays =  50 //int(rays_per_zone*(beam_max_z-beam_min_z)/dz) + 0;	// Can be used to launch more rays rather than just one.
//nrays=int(rays_per_zone*(beam_max_z-beam_min_z)/dz)+2;	// Can be used to launch more rays rather than just one.

/* print, "nrays per beam is", nrays */

/* Spatial dependence of laser power for an OMEGA sg5 phase plate is:
	exp(-1*((x/sigma)^2.0)^(5./2.))
	with sigma = 0.0375.  Need xmin/xmax ~ +/-0.0600		*/

/* This is the spatial dependence used by Russ Follett's CBET test problem. */
phase_x = span(beam_min_z,beam_max_z,nrays);
sigma = 1.7e-4; //2.0e-4;
pow_x = exp(-1*((phase_x/sigma)^2.0)^(4.0/2.0))

/*	Set the time step to be a less than dx/c, the Courant condition.
	Set the maximum number of time steps within the tracking function.
									*/
courant_mult = 0.2 // 0.37 // 0.25 // 0.36 // 0.22;
dt=courant_mult*min(dx,dz)/c;
nt=int(courant_mult^-1.0*max(nx,nz)*2.0);

/*	Define the x and z, k_x and k_z, and v_x and v_z arrays to store them for the ray.
	They are the positions, the wavevectors, and the group velocities
									*/
myx=array(0.0,nt);
myz=array(0.0,nt);
mykx=array(0.0,nt);
mykz=array(0.0,nt);
myvx=array(0.0,nt);
myvz=array(0.0,nt);
uray=array(1.0,nt);

finalts = array(0,nrays,nbeams);
mysaved_x = array(0.0,nt,nrays,nbeams);
mysaved_z = array(0.0,nt,nrays,nbeams);

/*	The bookkeeping array is named "marked". The first two sets of elements denote the zone number,
	labeled from 1 to (nx-1) and 1 to (nz-1).
	The third set of elements is the list of rays that passed through the zone.
														*/
numstored = int(5*rays_per_zone);		// maximum number of rays stored per zone
marked=array(0, nx, nz, numstored, nbeams);
present=array(0.0,nx,nz,nbeams);		// This array simply tallies the number of rays present in a
						// zone from each beam.

markingx = array(0,nt);
markingz = array(0,nt);
xbounds = array(0.0,nt);
zbounds = array(0.0,nt);
xbounds_double = array(0.0,nt);
zbounds_double = array(0.0,nt);

ncrossings = nx*3;	// Maximum number of potential grid crossings by a ray
crossesx = array(0.0,nbeams,nrays,ncrossings);
crossesz = array(0.0,nbeams,nrays,ncrossings);
boxes = array(0,nbeams,nrays,ncrossings,2);
ints = array(0.0,nbeams,nrays,ncrossings);

/* Set the initial location/position of the ray to be launched. */
x0 = array(0.0,nrays);
z0 = array(0.0,nrays);

offset = 0.5e-4				//offset = 0.0e-4

x0(1:nrays) = xmin-(dt/courant_mult*c*0.5);
z0(1:nrays) = span(beam_min_z,beam_max_z,nrays)+offset-(dz/2)-(dt/courant_mult*c*0.5);


/* Set the initial unnormalized k vectors, which give the initial direction
	of the launched ray.
   For example, kx = kz = 1 would give a 45 degree angle in the +x / +z direction.
   For example, kx = 0 (or kz = 0) would give pure kz (or kx) propagation. 	*/
kx0 = array(0.0,nrays);
kz0 = array(0.0,nrays);
kx0(1:nrays) = 1.0
kz0(1:nrays) = -0.1

intensity = 2.0e15                     // intensity of the beam in W/cm^2

uray_mult = intensity*(courant_mult)*double(rays_per_zone)^-1.0;

timer, elapsed, cat02;		// initialization timer category

/* Begin the loop over rays, to launch multiple rays in different directions
   from different initial locations. */

/* print,"Tracking rays..." */


beam=1;
/* print, "BEAMNUM is",beam; */
for(n=1;n<=nrays;++n){		// Loop over rays, this is only meaningful for nrays > 1
	raynum = n; // print,"raynum is",raynum;
	uray(1) = uray_mult*interp(pow_x,phase_x+offset,z0(n));	// Determines the initial power weighting

	dummy = launch_ray_XZ(x0(n),z0(n),kx0(n),kz0(n));
	finalts(n,beam) = finalt;
	mysaved_x(1:finalt,n,beam) = rayx;
	mysaved_z(1:finalt,n,beam) = rayz;
	/* if ( if_ray_tracker_diagnostic == 1 && n % 20 == 0){;
		print,"     ...",int(100.*(1.0-(double(n)/double(1*nrays)))),"%  remaining...";
	}; */
	timer, elapsed, cat07;	// plotting ray trajectories timer category
}	// End of nrays loop

// The following simply re-defines the x0,z0,kx0,kz0 for additional beams to launch.
z0(1:nrays) = zmin-(dt/courant_mult*c*0.5);
x0(1:nrays) = span(beam_min_z,beam_max_z,nrays)-(dx/2)-(dt/courant_mult*c*0.5)

kx0(1:nrays) = 0.0;
kz0(1:nrays) = 1.0;

beam=2;
/* print,"BEAMNUM is",beam; */
for(n=1;n<=nrays;++n){
	raynum = n; //  print,"raynum is",raynum;
	uray(1) = uray_mult*interp(pow_x,phase_x,x0(n));	// Determines the initial power weighting

    dummy = launch_ray_XZ(x0(n),z0(n),kx0(n),kz0(n));
	finalts(n,beam) = finalt;
	mysaved_x(1:finalt,n,beam) = rayx;
	mysaved_z(1:finalt,n,beam) = rayz;
	/* if ( if_ray_tracker_diagnostic == 1 && n % 20 == 0){;
		print,"     ...",int(100.*(1.0-(double(n)/double(1*nrays)))),"%  remaining...";
	}; */
	timer, elapsed, cat07;	// plotting ray trajectories timer cateogry
}       // End of nrays loop

/* Plot the cumulative energy deposited to the array edep,
	which shares the dimensions of x, z, eden, dedendz, etc.	*/

window,1;	// plots the ray trajectories
for (n=1;n<=nrays;++n){
    plg, mysaved_x(1:finalts(n,1), n, 1), mysaved_z(1:finalts(n,1), n, 1), marks=0, width=1,color="magenta";//"white"
    plg, mysaved_x(1:finalts(n,2), n, 2), mysaved_z(1:finalts(n,2), n, 2), marks=0, width=1,color="magenta";//"white"
}

window,2;
fma;
intensity_sum = edep(1:nx,1:nz,sum)			// edep(1:nx,1:nz,sum)
clo = 0.;
chi = max(intensity_sum);
plf,intensity_sum,x,z,cmin=clo,cmax=chi;
colorbar,clo,chi;
p9;
xytitles,"Z (cm)","X (cm)";
pltitle,"Overlapped intensity";

timer, elapsed, cat07;	// plotting ray trajectories timer cateogry


i_b1 = edep(1:nx,1:nz,1);
i_b2 = edep(1:nx,1:nz,2);

/* print,"Finding ray intersections with rays from opposing beams..." */

//========== FINDING AND SAVING ALL POINTS OF INTERSECTION BETWEEN THE BEAMS  =====================//

intersections = array(0.0,nx,nz);

/*	Loop through the spatial grid (2:nx,2:nz) to find where the rays from opposite beams intersect.
	The dimensions of the array are: marked( nx, nz, raynum, beamnum)
	// for raynum:  ss = 1; ss <= numstored; ++ss
	// for beamnum: bb = 1; bb <= nbeams; ++bb

	There will always be a beam #1, so start with beam 1 and check for rays present in the zone
	from other beams. Starting from beam #1, need to check for rays from beam #2, then #3, etc.
	After beam #1, need to start from beam #2 and check for rays from beam #3, then #4, etc.
	NOTE: Starting from beam #2, do NOT need to re-check for rays from beam #1.
*/

for (xx = 2; xx <= nx; ++xx){;			// The x and z loops start from 2, the first zone.
    for (zz = 2; zz <= nz; ++zz){;
        for (ss = 1; ss <= numstored; ++ss){;
            if ( marked(xx,zz,ss,1) == 0 ){;			// no ray present from beam 1
			    break;
            } else {;					// a ray is present from beam 1
	            for ( sss = 1 ; sss <= numstored  ; ++sss){;
	                if ( marked(xx,zz,sss,2) == 0 ){;	// no ray present from beam 2
	                    break;
                    } else {;		// a ray is also present from beam 2
                        intersections(xx,zz) += 1.0;
			        };	// end of if-else statement looking for specific rays from beam 2
				}; // end of sss loop looking for all rays from beam 2
			};	// end of if-else statement looking for ray from beam 1
		};	// end of ss loop
	};	// end of zz loop
};	// end of xx loop

timer, elapsed, cat09;	// finding intersections timer category


//========= CBET INTERACTIONS, GAIN COEFFICIENTS, AND ITERATION OF SOLUTION  =====================//

estat=4.80320427e-10; 	       // electron charge in statC
mach = -1.0*sqrt(2);                 // Mach number for max resonance
Z = 3.1;                        // ionization state
mi = 10230*(1.0e3*me);          // Mass of ion in g
mi_kg = 10230.0*me;	   // Mass of ion in kg
Te = 2.0e3*11604.5052;          // Temperature of electron in K
Te_eV = 2.0e3;
Ti = 1.0e3*11604.5052;          // Temperature of ion in K
Ti_eV = 1.0e3;
iaw = 0.2;                      // ion-acoustic wave energy-damping rate (nu_ia/omega_s)!!
kb = 1.3806485279e-16;   //Boltzmann constant in erg/K
kb2 = 1.3806485279e-23;   //Boltzmann constant in J/K

constant1 = (estat^2)/(4*(1.0e3*me)*c*omega*kb*Te*(1+3*Ti/(Z*Te)));

//====================CALCULATING CBET=================================//

/* print, "Calculating CBET gains..."; */

cs = 1e2*sqrt(ec*(Z*Te_eV+3.0*Ti_eV)/mi_kg);	// acoustic wave speed, approx. 4e7 cm/s in this example
u_flow = machnum*cs;    	// plasma flow velocity

// Find wavevectors, normalize, then multiply by magnitude.
// crossesx and crossesz have dimensions (nbeams, nrays, ncrossings)
dkx = crossesx(:,:,2:0)-crossesx(:,:,1:-1);
dkz = crossesz(:,:,2:0)-crossesz(:,:,1:-1);
dkmag = sqrt(dkx^2.0+dkz^2.0);

W1 = sqrt(1.0-eden/ncrit)/double(rays_per_zone);
W2 = sqrt(1.0-eden/ncrit)/double(rays_per_zone);

W1_init = W1;
W1_new = W1_init;
W2_init = W2;
W2_new = W2_init;

W1_storage = array(0.0,nx,nz,numstored);
W2_storage = array(0.0,nx,nz,numstored);

// The PROBE beam gains (loses) energy when gain2 < 0  (>0)
// The PUMP beam loses (gains) energy when gain2 < 0  (>0)

if_cbet_gain_diagnostics = 0;

for ( bb = 1; bb <= nbeams-1; ++bb){;
    for ( rr1 = 1; rr1 <= nrays; ++rr1){;
        for ( cc1 = 1; cc1 <= ncrossings; ++cc1){;
            if ( boxes(bb, rr1, cc1, 1) == 0 || boxes(bb, rr1, cc1, 2) == 0 ){;
				break;
            };
            ix = boxes(bb, rr1, cc1, 1);
            iz = boxes(bb, rr1, cc1, 2);

        	if ( intersections(ix,iz) != 0  ){;
				nonzeros1 = where(marked(ix,iz,,1));
				numrays1 = numberof(nonzeros1);
            	nonzeros2 = where(marked(ix,iz,,2));
				numrays2 = numberof(nonzeros2);

            	marker1 = marked(ix,iz,,1)(nonzeros1);
            	marker2 = marked(ix,iz,,2)(nonzeros2);
            	rr2 = marker2;
				cc2 = marker2;

            	for ( rrr = 1; rrr <= numrays1; ++rrr){;
                    if ( marker1(rrr) == rr1 ){;
                        ray1num = rrr;
                        break;
                    };
            	};
            	for ( n2 = 1; n2 <= numrays2; ++n2){;
                    for ( ccc = 1; ccc <= ncrossings; ++ccc){;
                        ix2 = boxes(bb+1, rr2(n2), ccc, 1);
                        iz2 = boxes(bb+1, rr2(n2), ccc, 2);
                        if ( ix == ix2 && iz == iz2 ){;
                                cc2(n2) = ccc;
                                break;
                        };
                    };
            	};

            	n2limit = min(present(ix,iz,1),numrays2);

            	for ( n2 = 1; n2 <= n2limit; ++n2){;
            		ne = eden(ix,iz);
            		epsilon = 1.0-ne/ncrit;
            		kmag = (omega/c)*sqrt(epsilon);         // magnitude of wavevector

            		kx1 = kmag * (dkx(bb, rr1, cc1) / (dkmag(bb, rr1, cc1) + 1.0e-10));

            		kx2 = kmag * (dkx(bb+1, rr2(n2), cc2(n2)) / (dkmag(bb+1, rr2(n2), cc2(n2)) + 1.0e-10));

            		kz1 = kmag * (dkz(bb, rr1, cc1) / (dkmag(bb, rr1, cc1) + 1.0e-10));

            		kz2 = kmag * (dkz(bb+1, rr2(n2), cc2(n2)) / (dkmag(bb+1, rr2(n2), cc2(n2)) + 1.0e-10));


            		kiaw = sqrt((kx2-kx1)^2.0+(kz2-kz1)^2.0);  // magnitude of the difference between the two vectors

            		ws = kiaw*cs            // acoustic frequency, cs is a constant
            		omega1= omega2= omega;  // laser frequency difference. To start, just zero.

					eta = ((omega2-omega1)-(kx2-kx1)*u_flow(ix,iz))/(ws+1.0e-10)


                	efield1 = sqrt(8.*pi*1.0e7*i_b1(ix,iz)/c);             // initial electric field of ray
                	efield2 = sqrt(8.*pi*1.0e7*i_b2(ix,iz)/c);             // initial electric field of ray

                	P = ((iaw)^2*eta)/((eta^2-1.0)^2+(iaw)^2*eta^2);         // From Russ's paper

            		gain1 = constant1*efield2^2*(ne/ncrit)*(1/iaw)*P;               //L^-1 from Russ's paper
            		gain2 = constant1*efield1^2*(ne/ncrit)*(1/iaw)*P;               //L^-1 from Russ's paper


            		// new energy of crossing (PROBE) ray (beam 2)
            		if ( dkmag(bb+1,rr2(n2),cc2(n2)) >= 1.0*dx ){;
                		W2_new(ix,iz) = W2(ix,iz)*exp(-1*W1(ix,iz)*dkmag(bb+1,rr2(n2),cc2(n2))*gain2/sqrt(epsilon));
                		W2_storage(ix,iz,n2) = W2_new(ix,iz);

                		// new energy of primary (PUMP) ray (beam 1)
						// USE W1_new formula:
                		W1_new(ix,iz) = W1(ix,iz)*exp(+1*W2(ix,iz)*dkmag(bb,rr1,cc1)*gain2/sqrt(epsilon));

						// ENFORCE Energy conservation:
						// W1_new(ix,iz) = W1(ix,iz)-(W2_new(ix,iz)-W2(ix,iz));

                		W1_storage(ix,iz,ray1num) = W1_new(ix,iz);
						// W1(ix,iz) = W1_new(ix,iz)
					};
           		}; // End of n2 loop for rays from beam 2 in the same box as the current ray from beam 1
    		}; // End of if-intersections check
        }; // End of cc loop
		/* if ( rr1 % 20 == 0 ) {;
			print,"     ...",int(100.*(1.0-(double(rr1)/double(1*nrays)))),"%  remaining...";
		}; */
    }; // End of rr loop
}; // End of bb loop

timer, elapsed, cat11;	// CBET calculations timer cateogry

/* clo = 0.99; chi= 1.0; */

/* print, "Updating intensities due to CBET gains..." */

i_b1_new = i_b1;
i_b2_new = i_b2;

for ( bb = 1; bb <= nbeams-1; ++bb){
    for ( rr1 = 1; rr1 <= nrays; ++rr1){
        for ( cc1 = 1; cc1 <= ncrossings; ++cc1){
            if ( boxes(bb, rr1, cc1, 1) == 0 || boxes(bb, rr1, cc1, 2) == 0 ){;
                break;
            };
            ix = boxes(bb, rr1, cc1, 1);
            iz = boxes(bb, rr1, cc1, 2);

			if ( intersections(ix,iz) != 0  ){;
                nonzeros1 = where(marked(ix,iz,,1));
				numrays1 = numberof(nonzeros1);
                nonzeros2 = where(marked(ix,iz,,2));
				numrays2 = numberof(nonzeros2);

                marker1 = marked(ix,iz,,1)(nonzeros1);
                marker2 = marked(ix,iz,,2)(nonzeros2);
                rr2 = marker2;
				cc2 = marker2;

                for ( rrr = 1; rrr <= numrays1; ++rrr){;
                    if ( marker1(rrr) == rr1 ){;
                        ray1num = rrr;
                        break;
                    };
                };
                for ( n2 = 1; n2 <= numrays2; ++n2){;
                    for ( ccc = 1; ccc <= ncrossings; ++ccc){;
                        ix2 = boxes(bb+1, rr2(n2), ccc, 1);
                        iz2 = boxes(bb+1, rr2(n2), ccc, 2);
                        if ( ix == ix2 && iz == iz2 ){;
                            cc2(n2) = ccc;
                            break;
                        };
                    };
                };

                fractional_change_1 = ( -1.0*( 1.0 - ( W1_new(ix,iz)/W1_init(ix,iz) ) ) * i_b1(ix,iz)  );
                fractional_change_2 = ( -1.0*( 1.0 - ( W2_new(ix,iz)/W2_init(ix,iz) ) ) * i_b2(ix,iz)  );

                i_b1_new(ix,iz) += fractional_change_1 ;
                i_b2_new(ix,iz) += fractional_change_2 ;

                x_prev_1 = x(ix,iz);
                z_prev_1 = z(ix,iz);
                x_prev_2 = x(ix,iz);
                z_prev_2 = z(ix,iz);

                // Now we need to find and increment/decrement the fractional_change for the rest of the beam 1 ray
                for ( ccc = cc1+1; ccc <= ncrossings; ++ccc){;
                    ix_next_1 = boxes(1, rr1, ccc, 1);
                    iz_next_1 = boxes(1, rr1, ccc, 2);

                    x_curr_1 = x(ix_next_1,iz_next_1);
                    z_curr_1 = z(ix_next_1,iz_next_1);

                    if ( ix_next_1 == 0 || iz_next_1 == 0 ){;
                        break;
                    } else {;
                        // Avoid double deposition if the (x,z) location doesn't change with incremented crossing number
                        if ( x_curr_1 != x_prev_1 || z_curr_1 != z_prev_1 ){;
                        	i_b1_new(ix_next_1,iz_next_1) += fractional_change_1 * (present(ix,iz,1)/present(ix_next_1,iz_next_1,1));
                    	};
                        x_prev_1 = x_curr_1;
                        z_prev_1 = z_curr_1;
                    };
                };	// End of increment/decreent for the rest of beam 1 ray

                n2 = min(ray1num,numrays2);		// In case numrays2 is less than the ray1num of the rays from beam 1 in the box.

                // Now we need to find and increment/decrement the fractional_change for the rest of the beam 2 ray
                for ( ccc = cc2(n2)+1; ccc <= ncrossings; ++ccc){;
                    ix_next_2 = boxes(2, rr2(n2), ccc, 1);
                    iz_next_2 = boxes(2, rr2(n2), ccc, 2);

                    x_curr_2 = x(ix_next_2,iz_next_2);
                    z_curr_2 = z(ix_next_2,iz_next_2);

                    if ( ix_next_2 == 0 || iz_next_2 == 0 ){;
                        break;
                    } else {;
                        // Avoid double deposition if the (x,z) location doesn't change with incremented crossing number
                        if ( x_curr_2 != x_prev_2 || z_curr_2 != z_prev_2 ){;
                        	i_b2_new(ix_next_2,iz_next_2) += fractional_change_2 * (present(ix,iz,1)/present(ix_next_2,iz_next_2,2));
                        };
	                    x_prev_2 = x_curr_2;
	                    z_prev_2 = z_curr_2;
                    };
                };	// End of increment/decrement for the rest of beam 2 ray
            }; // End of if-intersections check
		}; // End of cc loop

		/* if ( rr1 % 20 == 0 ){;
			print,"     ...",int(100.*(1.0-(double(rr1)/double(1*nrays)))),"%  remaining...";
		}; */
    }; // End of rr loop
}; // End of bb loop

timer, elapsed, cat11;	// CBET calculations timer cateogry

clo = 0.0e14; chi = 1.4*intensity;

window,3;
fma;
variable = 8.53e-10 * sqrt(i_b1 + i_b2 + 1.0e-10) * (1.053/3.0)

clo = 0;
chi = 0.021;
plf,variable,x,z,cmin=clo,cmax=chi; //plm, x, z;
//plfc,variable,x,z,levs=span(clo,chi,30);
colorbar,clo,chi;
p9;
xytitles,"Z (cm)","X (cm)";
pltitle,"Total original field amplitude (a0) ";

window,4;
fma;
variable = 8.53e-10*sqrt(max(1.0e-10,i_b1_new)+max(1.0e-10,i_b2_new))*(1.053/3.0)
clo = 0;
chi = 0.021;
plf,variable,x,z,cmin=clo,cmax=chi; //plm, x, z;
//plfc,variable,x,z,levs=span(clo,chi,60);
colorbar,clo,chi;
p9;
xytitles,"Z (cm)","X (cm)";
pltitle,"Total CBET new field amplitude (a0) ";

window,5;
fma;
a0variable = 8.53e-10*sqrt(max(1.0e-10,i_b1_new)+max(1.0e-10,i_b2_new))*(1.053/3.0)
//clo = 0; chi = 0.021;
plg, a0variable(:,2), x(:,1), marks=0, width=5, color="blue"
plg, a0variable(:,nz-1), x(:,1), marks=0, width=5, color="red"
plg, a0variable(:,(nz+1)/2), x(:,1), marks=0, width=5, color=[0,200,0]
xytitles,"X (cm)","a0";
pltitle," a0(x) at z_min_, z_0_, and z_max_";
gridxy,1,1

window,6;
fma;
a0variable = 8.53e-10*sqrt(max(1.0e-10,i_b1_new)+max(1.0e-10,i_b2_new))*(1.053/3.0)
//clo = 0; chi = 0.021;
plg, a0variable(2,:), z(1,:), marks=0, width=5, color="blue"
plg, a0variable(nx-1,:), z(1,:), marks=0, width=5, color="red"
plg, a0variable((nx+1)/2,:), z(1,:), marks=0, width=5, color=[0,200,0]
xytitles,"Z (cm)","a0";
pltitle," a0(z) at x_min_, x_0_, and x_max_";
gridxy,1,1



//==================== TIMER REPORTS =============================================================//

/* print,"FINISHED!   Reporting ray timings now...";
print,"___________________________________________________________________" */
timer, elapsed, cat10;
timer, elapsed0, total;

timer_print,
"Data import", cat01,
"Initialization", cat02,
"Initial ray index search", cat03,
"Index search in ray timeloop", cat04,
"Ray push", cat05,
"Mapping ray traj's to grid cat08", cat08,
"Mapping ray traj's to grid cat12", cat12,
"Interpolation for deposition", cat06,
"RAY LOOPS SUM", cat03+cat04+cat05+cat06+cat08,
"Finding intersections", cat09,
"Plotting ray information", cat07,
"CBET gain calculations", cat11,
"Others...", total-cat01-cat02-cat03-cat04-cat05-cat06-cat07-cat08-cat09-cat10-cat11-cat12,
"TOTAL", total;
/* print,"-------------------------------------------------------------------" */
