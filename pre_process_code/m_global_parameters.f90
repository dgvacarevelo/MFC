! MFC v3.0 - Pre-process Code: m_global_parameters.f90
! Description: This module contains all of the parameters characterizing the
!              computational domain, simulation algorithm, initial condition
!              and the stiffened equation of state.
! Author: Vedran Coralic
! Date: 06/10/12


MODULE m_global_parameters
    
    
    ! Dependencies =============================================================
    USE mpi                     ! Message passing interface (MPI) module
    
    USE m_derived_types        ! Definitions of the derived types
    ! ==========================================================================
    
    
    IMPLICIT NONE
    
    
    ! Logistics ================================================================
    INTEGER                    :: num_procs            ! Number of processors
    INTEGER        , PARAMETER :: num_stcls_min = 5    ! Mininum # of stencils
    INTEGER        , PARAMETER :: path_len      = 400  ! Maximum path length
    INTEGER        , PARAMETER :: name_len      = 50   ! Maximum name length
    REAL(KIND(0d0)), PARAMETER :: dflt_real     = -1d6 ! Default real value
    INTEGER        , PARAMETER :: dflt_int      = -100 ! Default integer value
    CHARACTER(LEN = path_len)  :: case_dir             ! Case folder location
    LOGICAL                    :: old_grid             ! Use existing grid data
    LOGICAL                    :: old_ic               ! Use existing IC data
    INTEGER                    :: t_step_old           ! Existing IC/grid folder
    ! ==========================================================================
    
    
    ! Computational Domain Parameters ==========================================
    
    ! Rank of the local processor
    INTEGER :: proc_rank
    
    ! Number of cells in the x-, y- and z-coordinate directions
    INTEGER :: m
    INTEGER :: n
    INTEGER :: p

    ! Global number of cells in each direction
    INTEGER :: m_glb, n_glb, p_glb

    ! Number of spatial dimensions
    INTEGER :: num_dims

    ! Cylindrical coordinates (either axisymmetric or full 3D)
    LOGICAL :: cyl_coord
    INTEGER :: grid_geometry

    ! Locations of cell-centers (cc) in x-, y- and z-directions, respectively
    REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:) :: x_cc, y_cc, z_cc

    ! Locations of cell-boundaries (cb) in x-, y- and z-directions, respectively
    REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:) :: x_cb, y_cb, z_cb
    
    ! Minimum cell-widths in the x-, y- and z-coordinate directions
    REAL(KIND(0d0)) :: dx, dy, dz
    
    ! Locations of the domain bounds in the x-, y- and z-coordinate directions
    TYPE(bounds_info) :: x_domain, y_domain, z_domain
    
    ! Grid stretching flags for the x-, y- and z-coordinate directions
    LOGICAL :: stretch_x, stretch_y, stretch_z
    
    ! Parameters of the grid stretching function for the x-, y- and z-coordinate
    ! directions. The "a" parameters are a measure of the rate at which the grid
    ! is stretched while the remaining parameters are indicative of the location
    ! on the grid at which the stretching begins.
    REAL(KIND(0d0)) :: a_x, a_y, a_z
    INTEGER         :: loops_x, loops_y, loops_z
    REAL(KIND(0d0)) :: x_a, y_a, z_a
    REAL(KIND(0d0)) :: x_b, y_b, z_b
    
    ! ==========================================================================
    
    
    ! Simulation Algorithm Parameters ==========================================
    INTEGER :: model_eqns      ! Multicomponent flow model
    INTEGER :: num_fluids      ! Number of different fluids present in the flow
    LOGICAL :: adv_alphan      ! Advection of the last volume fraction
    LOGICAL :: mpp_lim          ! Alpha limiter
    INTEGER :: sys_size        ! Number of unknowns in the system of equations
    INTEGER :: weno_order      ! Order of accuracy for the WENO reconstruction
    
    ! Annotations of the structure, i.e. the organization, of the state vectors
    TYPE(int_bounds_info) :: cont_idx                   ! Indexes of first & last continuity eqns.
    TYPE(int_bounds_info) :: mom_idx                    ! Indexes of first & last momentum eqns.
    INTEGER               :: E_idx                      ! Index of total energy equation
    INTEGER               :: alf_idx                    ! Index of void fraction
    TYPE(int_bounds_info) :: adv_idx                    ! Indexes of first & last advection eqns.
    TYPE(int_bounds_info) :: internalEnergies_idx       ! Indexes of first & last internal energy eqns.
    TYPE(bub_bounds_info) :: bub_idx                    ! Indexes of first & last bubble variable eqns.
    INTEGER               :: gamma_idx                  ! Index of specific heat ratio func. eqn.
    INTEGER               :: pi_inf_idx                 ! Index of liquid stiffness func. eqn.
    
    ! Boundary conditions in the x-, y- and z-coordinate directions
     TYPE(bounds_info) :: bc_x, bc_y, bc_z

    ! Format of the data files
    LOGICAL :: parallel_io
   
    ! Perturb density of surrounding air so as to break symmetry of grid
    LOGICAL :: perturb_flow
    INTEGER :: perturb_flow_fluid ! Fluid to be perturbed with perturb_flow flag
    LOGICAL :: perturb_sph
    INTEGER :: perturb_sph_fluid ! Fluid to be perturbed with perturb_sph flag
    REAL(KIND(0d0)), DIMENSION(num_fluids_max) :: fluid_rho

    ! Processor coordinates in MPI_CART_COMM
    INTEGER, ALLOCATABLE, DIMENSION(:) :: proc_coords
    ! Starting cell-center index of local processor in global grid
    INTEGER, ALLOCATABLE, DIMENSION(:) :: start_idx

    TYPE(mpi_io_var), PUBLIC :: MPI_IO_DATA

    ! MPI info for parallel IO with Lustre file systems
    CHARACTER(LEN = name_len) :: mpiiofs
    INTEGER :: mpi_info_int

    INTEGER, PRIVATE :: ierr
    ! ==========================================================================
    
    
    ! Initial Condition Parameters =============================================
    INTEGER :: num_patches     ! Number of patches composing initial condition
    
    ! Database of the initial condition patch parameters (icpp) for each of the
    ! patches employed in the configuration of the initial condition. Note that
    ! the maximum allowable number of patches, num_patches_max, may be changed
    ! in the module m_derived_types.f90.
    TYPE(ic_patch_parameters), DIMENSION(num_patches_max) :: patch_icpp
    ! ==========================================================================
    
    
    ! Fluids Physical Parameters ===============================================
    
    ! Database of the physical parameters of each of the fluids that is present
    ! in the flow. These include the stiffened gas equation of state parameters,
    ! the Reynolds numbers and the Weber numbers.
    TYPE(physical_parameters), DIMENSION(num_fluids_max) :: fluid_pp
    ! ==========================================================================
    
    ! SHB: Reference parameters for Tait EOS
    REAL(KIND(0d0)) :: rhoref, pref

    ! SHB: For bubble modeling
    INTEGER         :: nb
    REAL(KIND(0d0)) :: R0ref
    REAL(KIND(0d0)) :: Ca, Web, Re_inv
    REAL(KIND(0d0)), dimension(:), allocatable :: weight, R0, V0
    LOGICAL         :: bubbles


    ! SHB: For non-polytropic modeling
    LOGICAL         :: polytropic
    INTEGER         :: thermal  !1 = adiabatic, 2 = isotherm, 3 = transfer
    REAL(kind(0d0)) :: R_n, R_v, phi_vn, phi_nv, Pe_c, Tw
    REAL(kind(0d0)), dimension(:), allocatable :: k_n, k_v, pb0, mass_n0, mass_v0, Pe_T 
    REAL(kind(0d0)), dimension(:), allocatable :: Re_trans_T, Re_trans_c, Im_trans_T, Im_trans_c, omegaN 
    
    integer, allocatable, dimension(:,:,:) :: logic_grid
            


    ! Mathematical and Physical Constants ======================================
    REAL(KIND(0d0)), PARAMETER :: pi = 3.141592653589793d0
    ! ==========================================================================

    CONTAINS
        
        SUBROUTINE s_assign_default_values_to_user_inputs() ! ------------------
        ! Description: Assigns default values to user inputs prior to reading
        !              them in. This allows for an easier consistency check of
        !              these parameters once they are read from the input file.
            
            
            ! Generic loop iterator
            INTEGER :: i
            
            
            ! Logistics
            case_dir   = ' '
            old_grid   = .FALSE.
            old_ic     = .FALSE.
            t_step_old = dflt_int
            
            
            ! Computational domain parameters
            m = dflt_int
            n = dflt_int
            p = dflt_int
            
            cyl_coord = .FALSE.

            x_domain%beg = dflt_real
            x_domain%end = dflt_real
            y_domain%beg = dflt_real
            y_domain%end = dflt_real
            z_domain%beg = dflt_real
            z_domain%end = dflt_real
            
            stretch_x = .FALSE.
            stretch_y = .FALSE.
            stretch_z = .FALSE.
            
            a_x = dflt_real
            a_y = dflt_real
            a_z = dflt_real
            loops_x = 1 
            loops_y = 1 
            loops_z = 1 
            x_a = dflt_real
            x_b = dflt_real
            y_a = dflt_real
            y_b = dflt_real
            z_a = dflt_real
            z_b = dflt_real

            ! Simulation algorithm parameters
            model_eqns = dflt_int
            num_fluids = dflt_int
            adv_alphan = .FALSE.
            weno_order = dflt_int
            
            bc_x%beg = dflt_real
            bc_x%end = dflt_real
            bc_y%beg = dflt_real
            bc_y%end = dflt_real
            bc_z%beg = dflt_real
            bc_z%end = dflt_real
            
            parallel_io = .FALSE.
            perturb_flow = .FALSE.
            perturb_flow_fluid = dflt_int
            perturb_sph = .FALSE.
            perturb_sph_fluid = dflt_int
            fluid_rho = dflt_real
            
            
            ! Initial condition parameters
            num_patches = dflt_int
            
            DO i = 1, num_patches_max
                patch_icpp(i)%geometry        = dflt_int
                patch_icpp(i)%x_centroid      = dflt_real
                patch_icpp(i)%y_centroid      = dflt_real
                patch_icpp(i)%z_centroid      = dflt_real
                patch_icpp(i)%length_x        = dflt_real
                patch_icpp(i)%length_y        = dflt_real
                patch_icpp(i)%length_z        = dflt_real
                patch_icpp(i)%radius          = dflt_real
                patch_icpp(i)%epsilon         = dflt_real
                patch_icpp(i)%beta            = dflt_real
                patch_icpp(i)%normal          = dflt_real
                patch_icpp(i)%radii           = dflt_real
                patch_icpp(i)%alter_patch     = .FALSE.
                patch_icpp(i)%alter_patch(0)  = .TRUE.
                patch_icpp(i)%smoothen        = .FALSE.
                patch_icpp(i)%smooth_patch_id = i
                patch_icpp(i)%smooth_coeff    = dflt_real
                patch_icpp(i)%alpha_rho       = dflt_real
                patch_icpp(i)%rho             = dflt_real
                patch_icpp(i)%vel             = dflt_real
                patch_icpp(i)%pres            = dflt_real
                patch_icpp(i)%alpha           = dflt_real
                patch_icpp(i)%gamma           = dflt_real
                patch_icpp(i)%pi_inf          = dflt_real
                !should get all of r0's and v0's
                patch_icpp(i)%r0              = dflt_real
                patch_icpp(i)%v0              = dflt_real
            END DO
           
            !SHB Tait EOS
            rhoref  = dflt_real
            pref    = dflt_real

            !SHB Bubble modeling
            bubbles     = .FALSE.
            polytropic  = .TRUE.
            thermal     = dflt_int
            R0ref       = dflt_real
            nb          = dflt_int

            Ca      = dflt_real
            Re_inv  = dflt_real
            Web     = dflt_real

            R_n     = dflt_real
            R_v     = dflt_real
            phi_vn  = dflt_real
            phi_nv  = dflt_real
            Pe_c    = dflt_real
            Tw      = dflt_real

            ! Fluids physical parameters
            DO i = 1, num_fluids_max
                fluid_pp(i)%gamma   = dflt_real
                fluid_pp(i)%pi_inf  = dflt_real
                fluid_pp(i)%mul0    = dflt_real
                fluid_pp(i)%ss      = dflt_real
                fluid_pp(i)%pv      = dflt_real
                fluid_pp(i)%gamma_v = dflt_real
                fluid_pp(i)%M_v     = dflt_real
                fluid_pp(i)%mu_v    = dflt_real
                fluid_pp(i)%k_v     = dflt_real
            END DO
            
            
        END SUBROUTINE s_assign_default_values_to_user_inputs ! ----------------
        
        
        SUBROUTINE s_initialize_global_parameters_module() ! ----------------------
        ! Description: Computation of parameters, allocation procedures, and/or
        !              any other tasks needed to properly setup the module
            
            INTEGER :: i, fac
            
            ! Determining the layout of the state vectors and overall size of
            ! the system of equations, given the dimensionality and choice of
            ! the equations of motion
            
            ! Gamma/Pi_inf Model ===============================================
            IF(model_eqns == 1) THEN
                
                ! Setting number of fluids
                num_fluids = 1
                
                ! Annotating structure of the state and flux vectors belonging
                ! to the system of equations defined by the selected number of
                ! spatial dimensions and the gamma/pi_inf model
                cont_idx%beg = 1
                cont_idx%end = cont_idx%beg
                mom_idx%beg  = cont_idx%end + 1
                mom_idx%end  = cont_idx%end + num_dims
                E_idx        = mom_idx%end + 1
                adv_idx%beg  = E_idx + 1
                adv_idx%end  = adv_idx%beg + 1
                gamma_idx    = adv_idx%beg
                pi_inf_idx   = adv_idx%end
                sys_size     = adv_idx%end
                
            ! ==================================================================
                
                
            ! Volume Fraction Model (5-equation model) =========================
            ELSE IF(model_eqns == 2) THEN
                
                ! Annotating structure of the state and flux vectors belonging
                ! to the system of equations defined by the selected number of
                ! spatial dimensions and the volume fraction model
                cont_idx%beg = 1
                cont_idx%end = num_fluids
                mom_idx%beg  = cont_idx%end + 1
                mom_idx%end  = cont_idx%end + num_dims
                E_idx        = mom_idx%end + 1
                adv_idx%beg  = E_idx + 1
                adv_idx%end  = E_idx + num_fluids

                IF( (adv_alphan .NEQV. .TRUE.) .and. &
                        (num_fluids > 1)) adv_idx%end = adv_idx%end - 1
 
                sys_size = adv_idx%end

                IF (bubbles) THEN
                    alf_idx  = adv_idx%end
                ELSE
                    alf_idx = 0
                END IF
                
                if (bubbles) then
                    bub_idx%beg = sys_size+1
                    bub_idx%end = sys_size+2*nb
                    if (polytropic .neqv. .TRUE.) then
                        bub_idx%end = sys_size+4*nb
                    end if
                    sys_size = bub_idx%end

                    allocate( bub_idx%rs(nb), bub_idx%vs(nb) )
                    allocate( bub_idx%ps(nb), bub_idx%ms(nb) )
                    allocate( weight(nb),R0(nb),V0(nb) )

                    do i = 1, nb
                        if (polytropic .neqv. .TRUE.) then
                            fac = 4
                        else
                            fac = 2
                        end if
                        
                        bub_idx%rs(i) = bub_idx%beg+(i-1)*fac
                        bub_idx%vs(i) = bub_idx%rs(i)+1

                        if (polytropic .neqv. .TRUE.) then
                            bub_idx%ps(i) = bub_idx%vs(i)+1
                            bub_idx%ms(i) = bub_idx%ps(i)+1
                        end if
                    end do

                    if (nb == 1) then
                        weight(:)   = 1d0
                        R0(:)       = 1d0
                        V0(:)       = 0d0
                    else if (nb > 1) then
                        call s_simpson(nb)
                        V0(:)       = 0d0
                    else
                        stop 'Invalid value of nb'
                    end if

                    if (polytropic .neqv. .TRUE.) then
                        call s_initialize_nonpoly
                    else
                        rhoref  = 1.d0
                        pref    = 1.d0
                    end if
                end if                    
            ! ==================================================================


            ! Volume Fraction Model (6-equation model) =========================
            ELSE IF(model_eqns == 3) THEN

                ! Annotating structure of the state and flux vectors belonging
                ! to the system of equations defined by the selected number of
                ! spatial dimensions and the volume fraction model
                cont_idx%beg = 1
                cont_idx%end = num_fluids
                mom_idx%beg  = cont_idx%end + 1
                mom_idx%end  = cont_idx%end + num_dims
                E_idx        = mom_idx%end + 1
                adv_idx%beg  = E_idx + 1
                adv_idx%end  = E_idx + num_fluids
                IF(adv_alphan .NEQV. .TRUE.) adv_idx%end = adv_idx%end - 1
                internalEnergies_idx%beg  = adv_idx%end + 1
                internalEnergies_idx%end  = adv_idx%end + num_fluids
                sys_size     = internalEnergies_idx%end
            !========================
            ELSE IF(model_eqns == 4) THEN
                !SHB: 4 equation model with subgrid bubbles
                !only works with one fluid
                cont_idx%beg = 1 ! one continuity equation
                cont_idx%end = 1 ! num_fluids
                mom_idx%beg  = cont_idx%end + 1 ! one momentum equation in each direction
                mom_idx%end  = cont_idx%end + num_dims
                E_idx        = mom_idx%end + 1 ! one energy equation
                adv_idx%beg  = E_idx + 1
                adv_idx%end  = adv_idx%beg !one volume advection equation
                alf_idx      = adv_idx%end
                sys_size     = alf_idx !adv_idx%end

                if (bubbles) then
                    bub_idx%beg = sys_size+1
                    bub_idx%end = sys_size+2*nb
                    if (polytropic .neqv. .TRUE.) then
                        bub_idx%end = sys_size+4*nb
                    end if
                    sys_size = bub_idx%end
                    print*, 'sys_size', sys_size

                    allocate( bub_idx%rs(nb), bub_idx%vs(nb) )
                    allocate( bub_idx%ps(nb), bub_idx%ms(nb) )
                    allocate( weight(nb),R0(nb),V0(nb) )

                    do i = 1, nb
                        if (polytropic .neqv. .TRUE.) then
                            fac = 4
                        else
                            fac = 2
                        end if
                        
                        bub_idx%rs(i) = bub_idx%beg+(i-1)*fac
                        bub_idx%vs(i) = bub_idx%rs(i)+1

                        if (polytropic .neqv. .TRUE.) then
                            bub_idx%ps(i) = bub_idx%vs(i)+1
                            bub_idx%ms(i) = bub_idx%ps(i)+1
                        end if
                    end do

                    if (nb == 1) then
                        weight(:)   = 1d0
                        R0(:)       = 1d0
                        V0(:)       = 0d0
                    else if (nb > 1) then
                        call s_simpson(nb)
                        V0(:)       = 0d0
                    else
                        stop 'Invalid value of nb'
                    end if

                    if (polytropic .neqv. .TRUE.) then
                        call s_initialize_nonpoly
                    else
                        rhoref  = 1.d0
                        pref    = 1.d0
                    end if
                end if
            END IF
            ! ==================================================================

            ALLOCATE(MPI_IO_DATA%view(1:sys_size))
            ALLOCATE(MPI_IO_DATA%var(1:sys_size))

            DO i = 1, sys_size
                ALLOCATE(MPI_IO_DATA%var(i)%sf(0:m,0:n,0:p))
                MPI_IO_DATA%var(i)%sf => NULL()
            END DO

            ! Allocating grid variables for the x-direction
            ALLOCATE(x_cc(0:m), x_cb(-1:m))
            ! Allocating grid variables for the y- and z-directions
            IF(n > 0) THEN
                ALLOCATE(y_cc(0:n), y_cb(-1:n))
                IF(p > 0) THEN
                    ALLOCATE(z_cc(0:p), z_cb(-1:p))
                END IF
            END IF

            IF (cyl_coord .NEQV. .TRUE.) THEN ! Cartesian grid
                grid_geometry = 1
            ELSEIF (cyl_coord .AND. p == 0) THEN ! Axisymmetric cylindrical grid
                grid_geometry = 2
            ELSE ! Fully 3D cylindrical grid
                grid_geometry = 3
            END IF

            
            allocate( logic_grid(0:m,0:n,0:p) )
            
        END SUBROUTINE s_initialize_global_parameters_module ! --------------------
        
        
        subroutine s_initialize_nonpoly
            INTEGER :: ir
            REAL(KIND(0.D0)) :: rhol0
            REAL(KIND(0.D0)) :: pl0
            REAL(KIND(0.D0)) :: uu
            REAL(KIND(0.D0)) :: D_m
            REAL(KIND(0.D0)) :: temp
            REAL(KIND(0.D0)) :: omega_ref
            REAL(KIND(0.D0)), DIMENSION(Nb) :: chi_vw0
            REAL(KIND(0.D0)), DIMENSION(Nb) :: cp_m0
            REAL(KIND(0.D0)), DIMENSION(Nb) :: k_m0
            REAL(KIND(0.D0)), DIMENSION(Nb) :: rho_m0
            REAL(KIND(0.D0)), DIMENSION(Nb) :: x_vw
            ! polytropic index used to compute isothermal natural frequency
            REAL(KIND(0.D0)), PARAMETER :: k_poly = 1.D0
            ! universal gas constant
            REAL(KIND(0.D0)), PARAMETER :: Ru = 8314.D0


            ! liquid physical properties
            REAL(KIND(0.D0)) :: mul0, ss, pv, gamma_v, M_v, mu_v

            ! gas physical properties
            REAL(KIND(0.D0)) :: gamma_m, gamma_n, M_n, mu_n

            rhol0 = rhoref
            pl0   = pref
            
            allocate( pb0(nb), mass_n0(nb), mass_v0(nb), Pe_T(nb) )
            allocate( k_n(nb), k_v(nb), omegaN(nb) )
            allocate( Re_trans_T(nb), Re_trans_c(nb), Im_trans_T(nb), Im_trans_c(nb) ) 
            
            pb0(:)      = dflt_real
            mass_n0(:)  = dflt_real
            mass_v0(:)  = dflt_real
            Pe_T(:)     = dflt_real
            omegaN(:)   = dflt_real

            mul0    = fluid_pp(1)%mul0
            ss      = fluid_pp(1)%ss
            pv      = fluid_pp(1)%pv
            gamma_v = fluid_pp(1)%gamma_v
            M_v     = fluid_pp(1)%M_v
            mu_v    = fluid_pp(1)%mu_v
            k_v(:)  = fluid_pp(1)%k_v

            gamma_n = fluid_pp(2)%gamma_v
            M_n     = fluid_pp(2)%M_v
            mu_n    = fluid_pp(2)%mu_v
            k_n(:)  = fluid_pp(2)%k_v

            gamma_m = gamma_n
            if (thermal==2 ) gamma_m = 1.d0 !isothermal

            temp = 293.15D0
            D_m  = 0.242D-4
            uu = DSQRT( pl0/rhol0 )

            omega_ref = 3.D0*k_poly*Ca + 2.D0*( 3.D0*k_poly-1.D0 )/Web
           
            !!! thermal properties !!!
            ! gas constants
            R_n = Ru/M_n
            R_v = Ru/M_v
            ! phi_vn & phi_nv (phi_nn = phi_vv = 1)
            phi_vn = ( 1.D0+DSQRT(mu_v/mu_n)*(M_n/M_v)**(0.25D0) )**2 &
                / ( DSQRT(8.D0)*DSQRT(1.D0+M_v/M_n) )
            phi_nv = ( 1.D0+DSQRT(mu_n/mu_v)*(M_v/M_n)**(0.25D0) )**2 &
                / ( DSQRT(8.D0)*DSQRT(1.D0+M_n/M_v) )
            ! internal bubble pressure
            pb0 = pl0 + 2.D0*ss/( R0ref*R0 )            
            
            ! mass fraction of vapor
            chi_vw0 = 1.D0/( 1.D0+R_v/R_n*(pb0/pv-1.D0) )
            ! specific heat for gas/vapor mixture
            cp_m0 = chi_vw0*R_v*gamma_v/( gamma_v-1.D0 ) &
                  + ( 1.D0-chi_vw0 )*R_n*gamma_n/( gamma_n-1.D0 )
            ! mole fraction of vapor
            x_vw = M_n*chi_vw0/( M_v+(M_n-M_v)*chi_vw0 )
            ! thermal conductivity for gas/vapor mixture
            k_m0 = x_vw*k_v/( x_vw+(1.D0-x_vw)*phi_vn ) &
                 + ( 1.D0-x_vw )*k_n/( x_vw*phi_nv+1.D0-x_vw )
            ! mixture density
            rho_m0 = pv/( chi_vw0*R_v*temp )

            ! mass of gas/vapor computed using dimensional quantities
            mass_n0 = 4.D0*( pb0-pv )*pi/( 3.D0*R_n*temp*rhol0 )*R0**3
            mass_v0 = 4.D0*pv*pi/( 3.D0*R_v*temp*rhol0 )*R0**3
            ! Peclet numbers
            Pe_T = rho_m0*cp_m0*uu*R0ref/k_m0
            Pe_c = uu*R0ref/D_m
            ! nondimensional properties
            R_n = rhol0*R_n*temp/pl0
            R_v = rhol0*R_v*temp/pl0
            k_n = k_n/k_m0
            k_v = k_v/k_m0
            pb0 = pb0/pl0
            pv = pv/pl0

            ! bubble wall temperature, normalized by T0, in the liquid
            ! keeps a constant (cold liquid assumption)
            Tw = 1.D0
            ! natural frequencies
            omegaN = DSQRT( 3.D0*k_poly*Ca+2.D0*(3.D0*k_poly-1.D0)/(Web*R0) )/R0

            pl0 = 1.d0
            DO ir = 1,Nb
                CALL s_transcoeff( omegaN(ir)*R0(ir),Pe_T(ir)*R0(ir), &
                          Re_trans_T(ir),Im_trans_T(ir) )
                CALL s_transcoeff( omegaN(ir)*R0(ir),Pe_c*R0(ir), &
                          Re_trans_c(ir),Im_trans_c(ir) )
            END DO     
            Im_trans_T = 0d0
            Im_trans_c = 0d0

            rhoref = 1.d0
            pref = 1.d0
        end subroutine s_initialize_nonpoly
        
        SUBROUTINE s_transcoeff( omega,peclet,Re_trans,Im_trans )

            REAL(KIND(0.D0)), INTENT(IN) :: omega
            REAL(KIND(0.D0)), INTENT(IN) :: peclet
            REAL(KIND(0.D0)), INTENT(OUT) :: Re_trans
            REAL(KIND(0.D0)), INTENT(OUT) :: Im_trans
            COMPLEX :: trans, c1, c2, c3
            COMPLEX :: imag = ( 0.,1. )
            REAL(KIND(0.D0)) :: f_transcoeff

            c1 = imag*omega*peclet
            c2 = CSQRT( c1 )
            c3 = ( CEXP(c2)-CEXP(-c2) )/( CEXP(c2)+CEXP(-c2) ) ! TANH(c2)
            trans = ( (c2/c3-1.D0)**(-1)-3.D0/c1 )**( -1 ) ! transfer function

            Re_trans = DBLE( trans )
            Im_trans = AIMAG( trans )

        END SUBROUTINE s_transcoeff


        SUBROUTINE s_initialize_parallel_io() ! --------------------------------

            num_dims = 1 + MIN(1,n) + MIN(1,p)

            ALLOCATE(proc_coords(1:num_dims))

            IF (parallel_io .NEQV. .TRUE.) RETURN

            ! Option for Lustre file system (Darter/Comet/Stampede)
            WRITE(mpiiofs, '(A)') '/lustre_'
            mpiiofs = TRIM(mpiiofs)
            CALL MPI_INFO_CREATE(mpi_info_int, ierr)
            CALL MPI_INFO_SET(mpi_info_int, 'romio_ds_write', 'disable', ierr)

            ! Option for UNIX file system (Hooke/Thomson)
            ! WRITE(mpiiofs, '(A)') '/ufs_'
            ! mpiiofs = TRIM(mpiiofs)
            ! mpi_info_int = MPI_INFO_NULL
            
            ALLOCATE(start_idx(1:num_dims))

        END SUBROUTINE s_initialize_parallel_io ! ------------------------------


        SUBROUTINE s_finalize_global_parameters_module() ! ------------------------

            INTEGER :: i

            ! Deallocating grid variables for the x-direction
            DEALLOCATE(x_cc, x_cb)
            ! Deallocating grid variables for the y- and z-directions
            IF(n > 0) THEN
                DEALLOCATE(y_cc, y_cb)
                IF(p > 0) THEN
                    DEALLOCATE(z_cc, z_cb)
                END IF
            END IF

            DEALLOCATE(proc_coords)
            IF (parallel_io) THEN
                DEALLOCATE(start_idx)
                DO i = 1, sys_size
                    MPI_IO_DATA%var(i)%sf => NULL()
                END DO

                DEALLOCATE(MPI_IO_DATA%var)
                DEALLOCATE(MPI_IO_DATA%view)
            END IF

        END SUBROUTINE s_finalize_global_parameters_module ! ----------------------

        !bubble routines
        SUBROUTINE s_comp_n_from_cons( vftmp,nRtmp,ntmp )
            ! vftemp is \alpha, nRtemp is nR=n*R, ntmp is n
            ! compute n from \alpha and nR(:)
            REAL(KIND(0.D0)), INTENT(IN) :: vftmp
            REAL(KIND(0.D0)), DIMENSION(nb), INTENT(IN) :: nRtmp
            REAL(KIND(0.D0)), INTENT(OUT) :: ntmp
            REAL(KIND(0.D0)) :: nR3

            CALL s_quad( nRtmp**3,nR3 )  !returns itself if NR0 = 1
            ntmp = DSQRT( (4.d0*pi/3.d0)*nR3/vftmp )

        END SUBROUTINE s_comp_n_from_cons

        SUBROUTINE s_comp_n_from_prim( vftmp,Rtmp,ntmp )
            ! vftemp is \alpha, Rtemp is R, ntmp is n
            ! compute n from \alpha and R(:)
            REAL(KIND(0.D0)), INTENT(IN) :: vftmp
            REAL(KIND(0.D0)), DIMENSION(nb), INTENT(IN) :: Rtmp
            REAL(KIND(0.D0)), INTENT(OUT) :: ntmp
            REAL(KIND(0.D0)) :: R3

            CALL s_quad( Rtmp**3,R3 )  !returns itself if NR0 = 1
            ntmp = (3.d0/(4.d0*pi)) * vftmp/R3

        END SUBROUTINE s_comp_n_from_prim

        SUBROUTINE s_quad( func,mom )

            REAL(KIND(0.D0)), DIMENSION(nb), INTENT(IN) :: func
            REAL(KIND(0.D0)), INTENT(OUT) :: mom

            
            mom = DOT_PRODUCT( weight,func )

        END SUBROUTINE s_quad

        SUBROUTINE s_simpson( Npt )

            INTEGER, INTENT(IN) :: Npt
            INTEGER :: ir
            REAL(KIND(0.D0)) :: R0mn
            REAL(KIND(0.D0)) :: R0mx
            REAL(KIND(0.D0)) :: dphi
            REAL(KIND(0.D0)) :: tmp
            REAL(KIND(0.D0)) :: sd
            REAL(KIND(0.D0)), DIMENSION(Npt) :: phi

            ! nondiml. min. & max. initial radii for numerical quadrature
            !sd   = 0.05D0
            !R0mn = 0.75D0
            !R0mx = 1.3D0

            !sd   = 0.3D0
            !R0mn = 0.3D0
            !R0mx = 6.D0

            sd   = 0.7D0
            R0mn = 0.12D0
            R0mx = 150.D0
            
            ! phi = ln( R0 ) & return R0
            DO ir = 1,Npt
                phi(ir) = DLOG( R0mn ) &
                    + DBLE( ir-1 )*DLOG( R0mx/R0mn )/DBLE( Npt-1 )
                R0(ir) = DEXP( phi(ir) )
            END DO
            dphi = phi(2) - phi(1)

            ! weights for quadrature using Simpson's rule
            DO ir = 2,Npt-1
                ! Gaussian
                tmp = DEXP( -0.5D0*(phi(ir)/sd)**2 )/DSQRT( 2.D0*pi )/sd
                IF ( MOD(ir,2)==0 ) THEN
                    weight(ir) = tmp*4.D0*dphi/3.D0
                ELSE
                    weight(ir) = tmp*2.D0*dphi/3.D0
                END IF
            END DO
            
            tmp = DEXP( -0.5D0*(phi(1)/sd)**2 )/DSQRT( 2.D0*pi )/sd
            weight(1) = tmp*dphi/3.D0
            tmp = DEXP( -0.5D0*(phi(Npt)/sd)**2 )/DSQRT( 2.D0*pi )/sd
            weight(Npt) = tmp*dphi/3.D0
            !NR0beg = 1

            print*, 'weights = ', weight(:)
            print*, 'R0s = ', R0(:)

        END SUBROUTINE s_simpson



END MODULE m_global_parameters