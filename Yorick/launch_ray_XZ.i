/*==========================================================================================
                        2D RAY LAUNCHER AND TRACKER FUNCTION
                        Professor A. B. Sefkow

		The essence of the routine is to enforce this governing equation:
		w^2 = wpe(x,z)^2 + c^2*(kx^2+kz^2),
		i.e. kx^2+kz^2 = c^-2*(w^2-wpe(x,z)^2),
		where vg = dw/dk = c^2*k/w = (c/w)*sqrt(omega^2-wpe(x,z)^2)

		And we update the ray positions (x) and group velocities (vg) according to
		d(x)/dt = vg, and
		d(vg)/dt = -c^2/2*(gradient(eden)/ncrit)

============================================================================================*/

func launch_ray_XZ(x_init,z_init,kx_init,kz_init){
	extern rayx, rayz, finalt, amp_norm, time;
	/* 	(x0,z0) is the initial starting point for the ray
		(kx0,kz0) is the initial wavevector for the ray
		In ray.i, these were myx(1),myz(1),kx_init, and kz_init	*/

		/* Set the initial location/position of the ray to be launched. */
		/* The (1) is used because it is the first (time) step, which we define (initial condition) */
	myx(1) = x_init;
	myz(1) = z_init;

	/* The following code will find the logical indices of the spatial locations on the mesh
    which correspond to thomyse initial positions. */
	for(xx=1;xx<=nx;++xx){
		if ( myx(1) - x(xx,1) <= (0.5+1.0e-10)*dx && myx(1) - x(xx,1) >= -(0.5+1.0e-10)*dx ){
	        thisx_0 = xx;
	        thisx_00 = xx;
	        break;  // "breaks" out of the xx loop once the if statement condition is met.
	    };
	};

	for(zz=1;zz<=nz;++zz){
    	if ( myz(1) - z(1,zz) <= (0.5+1.0e-10)*dz && myz(1) - z(1,zz) >= -(0.5+1.0e-10)*dz ){
        	thisz_0 = zz;
        	thisz_00 = zz;
        	break;  // "breaks" out of the zz loop once the if statement condition is met.
    	};
	};

	/* The (xi, zi) logical indices for the starting position are now (thisx_0, thisz_0). */
	timer, elapsed, cat03;          // first ray index search timer category

	/* Calculate the total k (=sqrt(kx^2+kz^2)) from the dispersion relation,
        taking into account the local plasma frequency of where the ray starts. */
	k = sqrt((omega^2.0 - wpe(thisx_0, thisz_0)^2.0) / c^2.0)

	/* Set the initial unnormalized k vectors, which give the initial direction
    of the launched ray.
   	For example, kx = kz = 1 would give a 45 degree angle in the +x / +z direction.
   	For example, kx = 0 (or kz = 0) would give pure kz (or kx) propagation.      */

	knorm=sqrt(kx_init^2.0+kz_init^2.0);		// Length of k for the ray to be launched
	mykx(1)=(kx_init/knorm)*k;			// Normalized value for the ray's initial k_x
	mykz(1)=(kz_init/knorm)*k;			// Normalized value for the ray's initial k_z
	myvx(1) = c^2.0*mykx(1)/omega;                   // v_group, group velocity (dw/dk) from D(k,w).
	myvz(1) = c^2.0*mykz(1)/omega;

	markingx(1) = thisx_0;
	markingz(1) = thisz_0;
	xbounds(1) = myx(1);
	zbounds(1) = myz(1);

	numcrossing = 1

	timer, elapsed, cat02;          // initialization timer category

	/* Begin the loop in time, where dt is the time step and each ray will be advanced in x and vg */
	for(tt = 2; tt <= nt; ++tt){                          // Time step loop

        /* We update the group velocities using grad(n_e), and then the positions
        by doing this sequence :
        v_x = v_x0 + -c^2/(2*nc) * d[ne(x)]/dx * dt
        v_z = v_z0 + -c^2/(2*nc) * d[ne(z)]/dz * dt
        x_x = x_x0 + v_x * dt
        x_z = x_z0 + v_z * dt*/

        myvz(tt) = myvz(tt-1) - c^2.0/(2.0*ncrit)*dedendz(thisx_0,thisz_0)*dt
        myvx(tt) = myvx(tt-1) - c^2.0/(2.0*ncrit)*dedendx(thisx_0,thisz_0)*dt
        myx(tt) = myx(tt-1) + myvx(tt)*dt
        myz(tt) = myz(tt-1) + myvz(tt)*dt

		timer, elapsed, cat05;	// ray push timer category

		//============== Update the index, and track the intersections and boxes ================

		search_index_x = 1	// nx	// Use nx for original (whole mesh) search
		search_index_z = 1	// nz	// Use nz for original (whole mesh) search
        thisx_m = max(1,thisx_0-search_index_x);
        thisx_p = min(nx,thisx_0+search_index_x);
        thisz_m = max(1,thisz_0-search_index_z);
        thisz_p = min(nz,thisz_0+search_index_z);

        for(xx = thisx_m; xx <= thisx_p; ++xx){                  // Determines current x index for the position
			if((myx(tt) - x(xx,1) <= (0.5+1.0e-10) * dx) && (myx(tt) - x(xx,1) >= -(0.5+1.0e-10)*dx)){
                thisx = xx;
                break;  // "breaks" out of the xx loop once the if statement condition is met.
            };
		};


        for(zz = thisz_m; zz <= thisz_p; ++zz){;                  // Determines current z index for the position
			if((myz(tt) - z(1,zz) <= (0.5+1.0e-10) * dz) && (myz(tt) - z(1,zz) >= -(0.5+1.0e-10)*dz)){
                thisz = zz;
                break;  // "breaks" out of the zz loop once the if statement condition is met.
            };
		};

        /* The (xi, zi) logical indices for the position are now stored as (thisx, thisz).
           We also need to reset thisx_0 and thisz_0 to the new indices thisx and thisz.   */
//        thisx_0 = thisx; thisz_0 = thisz;

        timer, elapsed, cat04;  // second index search for ray time loop timer category

		linez = [myz(tt-1),myz(tt)]             // In each loop, we analyze the current point and the point before
		linex = [myx(tt-1),myx(tt)]
		lastx = 10000           // An initialization, more details below
		lastz = 10000           // An initialization, more details below

        thisx_m = max(1,thisx_0-search_index_x);
        thisx_p = min(nx,thisx_0+search_index_x);
        thisz_m = max(1,thisz_0-search_index_z);
        thisz_p = min(nz,thisz_0+search_index_z);

        for(xx = thisx_m; xx <= thisx_p; ++xx){                  // Determines current x index for the position
			currx = (x(xx,1) - dx/2);
            if ( ( myx(tt) > currx && myx(tt-1) <= currx) || ( myx(tt) < currx && myx(tt-1) >= currx ) ){;
                crossx = interp(linez,linex,currx);         // find the Z point of intersection
				if (abs(crossx - lastz) > 1.0e-20 ){;
					/* ints(beam,raynum,numcrossing) = uray(tt); */
					crossesx(beam,raynum,numcrossing) = currx;
					crossesz(beam,raynum,numcrossing) = crossx;
					if ( myx(tt) <= xmax+dx/2 && myx(tt) >= xmin-dx/2 ){;
						boxes(beam,raynum,numcrossing,) = [thisx,thisz];
					};
                    lastx = currx;                  /* This keeps track of the last x-value added,
                                                        so that we don't double-count the corners */
					numcrossing += 1;
                    break;
				};
			};
        };

        for(zz = thisz_m; zz <= thisz_p; ++zz){;                  // Determines current z index for the position
            currz = z(1,zz) - (dz / 2);
            if((myz(tt) > currz && myz(tt-1) <= currz ) || ( myz(tt) < currz && myz(tt-1) >= currz)){;
                crossz = interp(linex,linez,currz);
                if ( abs(crossz - lastx) > 1.0e-20 ){;
					/* ints(beam,raynum,numcrossing) = uray(tt); */
					crossesx(beam,raynum,numcrossing) = crossz;
					crossesz(beam,raynum,numcrossing) = currz;
					if ( myz(tt) <= zmax+dz/2 && myz(tt) >= zmin-dz/2 ){;
						boxes(beam,raynum,numcrossing,) = [thisx,thisz];
					};
                    lastz = currz;                  /* This keeps track of the last z-value added,
                                                    	so that we don't double-count the corners */
					numcrossing += 1;
                    break;
                };
            };
        };

        thisx_0 = thisx;
		thisz_0 = thisz;

		timer, elapsed, cat08;	// mapping ray trajectories to grid timer category

        markingx(tt) = thisx;
		markingz(tt) = thisz;

        if ( markingx(tt) != markingx(tt-1) && markingz(tt) != markingz(tt-1) ){
            if ( myvz(tt) < 0.0 ){;
                ztarg = z(thisx,thisz)+(dz/2.0);
            } else if ( myvz(tt) >= 0.0 ){
                ztarg = z(thisx,thisz)-(dz/2.0);
            };
            slope = (myz(tt)-myz(tt-1))/(myx(tt)-myx(tt-1)+1.0e-10);
            xtarg = myx(tt-1)+(ztarg-myz(tt-1))/slope;
            xbounds(tt) = xtarg;                      // Saving the Z line crossing into x/zbounds
            zbounds(tt) = ztarg;

            if ( myvx(tt) >= 0.0 ){
                xtarg = x(thisx,thisz)-(dx/2.0);
            } else if ( myvx(tt) < 0.0 ) {
                xtarg = x(thisx,thisz)+(dx/2.0);
            }
            slope = (myx(tt)-myx(tt-1))/(myz(tt)-myz(tt-1)+1.0e-10);
            ztarg = myz(tt-1)+(xtarg-myx(tt-1))/slope;
            xbounds_double(tt) = xtarg               // Saving the X line crossing into x/zbounds_double
            zbounds_double(tt) = ztarg

            for (ss = 1; ss <= numstored; ++ss){
                if ( marked(thisx-0,thisz-0,ss,beam) == 0 ){
                    marked(thisx-0,thisz-0,ss,beam) = raynum
					present(thisx,thisz,beam) += 1.0;
                    break;
                }
            }
        } else if ( markingz(tt) != markingz(tt-1) ){
            if ( myvz(tt) < 0.0 ){
                ztarg = z(thisx,thisz)+(dz/2.0);
            } else if ( myvz(tt) >= 0.0 ){
                ztarg = z(thisx,thisz)-(dz/2.0);
            }
            slope = (myz(tt)-myz(tt-1))/(myx(tt)-myx(tt-1)+1.0e-10); // print,"slope is", slope;
            xtarg = myx(tt-1)+(ztarg-myz(tt-1))/slope;
            xbounds(tt) = xtarg
            zbounds(tt) = ztarg

            for (ss = 1; ss <= numstored; ++ss){
                if ( marked(thisx-0,thisz-0,ss,beam) == 0 ){
                    marked(thisx-0,thisz-0,ss,beam) = raynum
					present(thisx,thisz,beam) += 1.0;
                    break;
                }
            }
        } else if ( markingx(tt) != markingx(tt-1) ){
            if ( myvx(tt) >= 0.0 ){
                xtarg = x(thisx,thisz)-(dx/2.0);
            } else if ( myvx(tt) < 0.0 ) {
                xtarg = x(thisx,thisz)+(dx/2.0);
            }
            slope = (myx(tt)-myx(tt-1))/(myz(tt)-myz(tt-1)+1.0e-10); // print,"slope is", slope;
            ztarg = myz(tt-1)+(xtarg-myx(tt-1))/slope;
            xbounds(tt) = xtarg
            zbounds(tt) = ztarg

            for (ss = 1; ss <= numstored; ++ss){
                if ( marked(thisx-0,thisz-0,ss,beam) == 0 ){
                    marked(thisx-0,thisz-0,ss,beam) = raynum
					present(thisx,thisz,beam) += 1.0;
                    break;
                }
            }
        }

        timer, elapsed, cat12;  // mapping ray trajectories to grid timer category


		/* In order to calculate the deposited energy into the plasma,
		we need to calculate the plasma resistivity (eta) and collision frequency (nu_e-i) */

//		eta = 5.2e-5*10.0/(etemp(thisx,thisz)^1.5)		// ohm*m
//		nuei(tt) = (1e6*eden(thisx,thisz)*ec^2.0/me)*eta

		/* Now we can decrement the ray's energy density according to how much energy
		was absorbed by the plasma. */

//		uray(tt) = uray(tt-1) - (nuei(tt)*(eden(thisx,thisz)/ncrit)*uray(tt-1)*dt)
//		increment = (nuei(tt)*(eden(thisx,thisz)/ncrit)*uray(tt-1)*dt)

		/* We use these next two lines instead, if we are just using uray as a bookkeeping device
		(i.e., no absorption by the plasma and no loss of energy by the ray).	*/
		uray(tt) = uray(tt-1);
		increment = uray(tt);

		/* Rather than put all the energy into the cell in which the ray resides, which
		is the so-called "nearest-neighbor" approach (which is very noise and less accurate),
		we will use an area-based linear weighting scheme to deposit the energy to the
		four nearest nodes of the ray's current location. In 3D, this would be 8 nearest.   */

		// Define xp and zp to be the ray's position relative to the nearest node.
		xp = (myx(tt) - (x(thisx,thisz)+dx/2.0))/dx
		zp = (myz(tt) - (z(thisx,thisz)+dz/2.0))/dz


		/* f = open("xp_zp.csv", "a")
		write, f, format="%d, %.10f, %.10f\n", tt, xp, zp
		close, f */

		/*	Below, we interpolate the energy deposition to the grid using linear area weighting.
		The edep array must be two larger in each direction (one for min, one for max)
		to accomodate this routine, since it deposits energy in adjacent cells.		*/

		if ( xp >= 0 && zp >= 0 ){
			dl = zp;
			dm = xp;
			a1 = (1.0-dl)*(1.0-dm);		// blue		: (x  , z  )
			a2 = (1.0-dl)*dm;		// green	: (x+1, z  )
			a3 = dl*(1.0-dm);		// yellow	: (x  , z+1)
			a4 = dl*dm;			// red 		: (x+1, z+1)
			edep(thisx+1,thisz+1,beam) += a1*increment	// blue
			edep(thisx+1+1,thisz+1,beam) += a2*increment	// green
			edep(thisx+1,thisz+1+1,beam) += a3*increment	// yellow
			edep(thisx+1+1,thisz+1+1,beam) += a4*increment	// red
		} else if ( xp < 0 && zp >= 0 ){
			dl = zp;
			dm = abs(xp);		// because xp < 0
			a1 = (1.0-dl)*(1.0-dm);		// blue		: (x  , z  )
			a2 = (1.0-dl)*dm;		// green	: (x-1, z  )
			a3 = dl*(1.0-dm);		// yellow	: (x  , z+1)
			a4 = dl*dm;			// red 		: (x-1, z+1)
			edep(thisx+1,thisz+1,beam) += a1*increment	// blue
			edep(thisx+1-1,thisz+1,beam) += a2*increment	// green
			edep(thisx+1,thisz+1+1,beam) += a3*increment	// yellow
			edep(thisx+1-1,thisz+1+1,beam) += a4*increment	// red
		} else if ( xp >= 0 && zp < 0 ){
			dl = abs(zp);		// because zp < 0
			dm = xp;
			a1 = (1.0-dl)*(1.0-dm);		// blue		: (x  , z  )
			a2 = (1.0-dl)*dm;		// green	: (x+1, z  )
			a3 = dl*(1.0-dm);		// yellow	: (x  , z-1)
			a4 = dl*dm;			// red 		: (x+1, z-1)
			edep(thisx+1,thisz+1,beam) += a1*increment	// blue
			edep(thisx+1+1,thisz+1,beam) += a2*increment	// green
			edep(thisx+1,thisz+1-1,beam) += a3*increment	// yellow
			edep(thisx+1+1,thisz+1-1,beam) += a4*increment	// red
		} else if ( xp < 0 && zp < 0 ){
			dl = abs(zp);		// because zp < 0
			dm = abs(xp);		// because xp < 0
			a1 = (1.0-dl)*(1.0-dm);		// blue		: (x  , z  )
			a2 = (1.0-dl)*dm;		// green	: (x-1, z  )
			a3 = dl*(1.0-dm);		// yellow	: (x  , z-1)
			a4 = dl*dm;			// red 		: (x-1, z-1)
			edep(thisx+1,thisz+1,beam) += a1*increment	// blue
			edep(thisx+1-1,thisz+1,beam) += a2*increment	// green
			edep(thisx+1,thisz+1-1,beam) += a3*increment	// yellow
			edep(thisx+1-1,thisz+1-1,beam) += a4*increment	// red
		} else {
			print, "xp is", xp, "zp is", zp
			edep(thisx,thisz,beam) += (nuei(tt)*(eden(thisx,thisz)/ncrit)*uray(tt-1)*dt)
			print,"***** ERROR in interpolation of laser deposition to grid!! *****"
			break;
		}

		timer, elapsed, cat06;		// linear interpolation for deposition timer category

        /* This is how the amplitude of the E field is changed during propagation, but is not derived here.*/
        /* It assumes conservation of wave action. */

        // This will cause the code to stop following the ray once it escapes the extent of the plasma:
        if ( myx(tt) < (xmin-(dx/2.0)) | myx(tt) > (xmax+(dx/2.0))){          // the "|" means "or" (symbol above the return key)
            finalt = tt-1;

			rayx = myx(1:finalt);
			rayz = myz(1:finalt);

			timer, elapsed, cat02;          // initialization timer category
            break;                  // "breaks" out of the tt loop once the if condition is satisfied
        } else if ( myz(tt) < (zmin-(dz/2.0)) | myz(tt) > (zmax+(dz/2.0))){   // the "|" means "or" (symbol above the return key)
            finalt = tt-1;

			rayx = myx(1:finalt);
			rayz = myz(1:finalt);

			timer, elapsed, cat02;          // initialization timer category
            break;                  // "breaks" out of the tt loop once the if condition is satisfied
        }
	}       // End of time step loop for a single ray
}	// End of launch_ray_XZ function
