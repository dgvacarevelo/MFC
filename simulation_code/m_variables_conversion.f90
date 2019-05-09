! MFC v3.0 - Simulation Code: m_variables_conversion.f90
! Description: This module features a database of subroutines that allow for the
!              conversion of state variables from one type into another. At this
!              time, the state variables type conversions below are available:
!                             1) Mixture        => Mixture
!                             2) Species        => Mixture
!                             3) Conservative   => Primitive
!                             4) Conservative   => Characteristic
!                             5) Conservative   => Flux
!                             6) Primitive      => Conservative
!                             7) Primitive      => Characteristic
!                             8) Primitive      => Flux
!                             9) Characteristic => Conservative
!                            10) Characteristic => Primitive
! Author: Vedran Coralic
! Date: 07/29/12


MODULE m_variables_conversion
    
    
    ! Dependencies =============================================================
    USE m_derived_types        ! Definitions of the derived types
    
    USE m_global_parameters    ! Definitions of the global parameters
    
    USE m_mpi_proxy            ! Message passing interface (MPI) module proxy
    ! ==========================================================================
    
    
    IMPLICIT NONE
    
    PRIVATE; PUBLIC :: s_initialize_variables_conversion_module          , &
                       s_convert_to_mixture_variables                    , &
                       s_convert_mixture_to_mixture_variables            , &
                       s_convert_species_to_mixture_variables_bubbles    , &
                       s_convert_species_to_mixture_variables            , &
                       s_convert_conservative_to_primitive_variables     , &
                       s_convert_conservative_to_characteristic_variables, &
                       s_convert_conservative_to_flux_variables          , &
                       s_convert_primitive_to_conservative_variables     , &
                       s_convert_primitive_to_characteristic_variables   , &
                       s_convert_primitive_to_flux_variables             , &
                       s_convert_primitive_to_flux_variables_bubbles     , &
                       s_convert_characteristic_to_conservative_variables, &
                       s_convert_characteristic_to_primitive_variables   , &
                       s_compute_lsq_gradient_curvature                  , &
                       s_finalize_variables_conversion_module
    
    
    ABSTRACT INTERFACE ! =======================================================
        
        ! The abstract interface to the procedures that are utilized to convert
        ! the mixture and the species variables into the mixture variables. For
        ! more information, refer to:
        !               1) s_convert_mixture_to_mixture_variables
        !               2) s_convert_species_to_mixture_variables
        SUBROUTINE s_convert_abstract_to_mixture_variables( qK_vf, rho_K,      &
                                                            gamma_K, pi_inf_K, &
                                                            Re_K, We_K, i,j,k  )
            
            IMPORT :: scalar_field, sys_size, num_fluids
            
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: qK_vf
            
            REAL(KIND(0d0)), INTENT(OUT) :: rho_K, gamma_K, pi_inf_K
            
            REAL(KIND(0d0)), DIMENSION(2), INTENT(OUT) :: Re_K
            
            REAL(KIND(0d0)), DIMENSION( num_fluids, &
                                        num_fluids  ), INTENT(OUT) :: We_K
            
            INTEGER, INTENT(IN) :: i,j,k
            
        END SUBROUTINE s_convert_abstract_to_mixture_variables
        
        ! The abstract interface to the procedures that are used to compute the
        ! Roe and the arithmetic average states. For additional information see:
        !                 1) s_compute_roe_average_state
        !                 2) s_compute_arithmetic_average_state
        SUBROUTINE s_compute_abstract_average_state(i,j,k)
            
            INTEGER, INTENT(IN) :: i,j,k
            
        END SUBROUTINE s_compute_abstract_average_state
        
    END INTERFACE ! ============================================================
    
    
    ! The left (L) and right (R) cell-average densities, velocities, pressures,
    ! energies, enthalpies, mass fractions, the specific heat ratio and liquid
    ! stiffness functions, the shear and volume Reynolds numbers and the Weber
    ! numbers, respectively. These are defined with respect to cell-boundaries.
    REAL(KIND(0d0))                              ::    rho_L,    rho_R
    REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:)   ::    vel_L,    vel_R
    REAL(KIND(0d0))                              ::   pres_L,   pres_R
    REAL(KIND(0d0))                              ::      E_L,      E_R
    REAL(KIND(0d0))                              ::      H_L,      H_R
    REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:)   ::     mf_L,     mf_R
    REAL(KIND(0d0))                              ::  gamma_L,  gamma_R
    REAL(KIND(0d0))                              :: pi_inf_L, pi_inf_R
    REAL(KIND(0d0)),              DIMENSION(2)   ::     Re_L,     Re_R
    REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:,:) ::     We_L,     We_R
    ! SHB: alpha_L/R for mixture speed of sound computation    
    REAL(KIND(0d0))                              ::   alpha_L,   alpha_R
    
    ! The Roe or arithmetic average density, velocity, enthalpy, mass fractions,
    ! specific heat ratio function, and the sound speed, at the cell-boundaries,
    ! computed from the left and right cell-average variables.
    REAL(KIND(0d0))   , ALLOCATABLE, DIMENSION(:,:,:) :: rho_avg_sf
    REAL(KIND(0d0))   , ALLOCATABLE, DIMENSION(:)     :: vel_avg
    REAL(KIND(0d0))                                   :: H_avg
    TYPE(scalar_field), ALLOCATABLE, DIMENSION(:)     :: mf_avg_vf
    REAL(KIND(0d0))                                   :: gamma_avg
    REAL(KIND(0d0))   , ALLOCATABLE, DIMENSION(:,:,:) :: c_avg_sf
    ! SHB: alpha_avg for mixture SOS computation
    REAL(KIND(0d0))                                   :: alpha_avg
    REAL(KIND(0d0))                                   :: pres_avg
    
    ! Pointer to the procedure utilized to convert either the mixture or the
    ! species variables into the mixture variables, based on model equations
    PROCEDURE(s_convert_abstract_to_mixture_variables), &
    POINTER :: s_convert_to_mixture_variables => NULL()
    
    ! Pointer to the subroutine utilized to calculate either the Roe or the
    ! arithmetic average state variables, based on the chosen average state
    PROCEDURE(s_compute_abstract_average_state), &
    POINTER :: s_compute_average_state => NULL()
    
    
    CONTAINS
                
        SUBROUTINE s_convert_mixture_to_mixture_variables( qK_vf, rho_K,      &
                                                           gamma_K, pi_inf_K, &
                                                           Re_K, We_K, i,j,k  )
        ! Description: This procedure is used alongside with the gamma/pi_inf
        !              model to transfer the density, the specific heat ratio
        !              function and liquid stiffness function from the vector
        !              of conservative or primitive variables to their scalar
        !              counterparts.
            
            
            ! Conservative or primitive variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: qK_vf
            
            ! Density and the specific heat ratio and liquid stiffness functions
            REAL(KIND(0d0)), INTENT(OUT) :: rho_K, gamma_K, pi_inf_K
            
            ! Shear and volume Reynolds numbers
            REAL(KIND(0d0)), DIMENSION(2), INTENT(OUT) :: Re_K
            
            ! Weber numbers
            REAL(KIND(0d0)), DIMENSION( num_fluids, &
                                        num_fluids  ), INTENT(OUT) :: We_K
            
            ! Generic loop iterators
            INTEGER, INTENT(IN) :: i,j,k
            
            ! Performing the transfer of the density, the specific heat ratio
            ! function as well as the liquid stiffness function, respectively
            rho_K    = qK_vf(1)%sf(i,j,k)
            gamma_K  = qK_vf(gamma_idx)%sf(i,j,k)
            pi_inf_K = qK_vf(pi_inf_idx)%sf(i,j,k)
            
        END SUBROUTINE s_convert_mixture_to_mixture_variables ! ----------------
       
        !!! SHB mixture variables if you have one phase plus modeled bubbles
        SUBROUTINE s_convert_species_to_mixture_variables_bubbles ( qK_vf, rho_K,      &
                                                           gamma_K, pi_inf_K, &
                                                           Re_K, We_K, i,j,k  )

        ! Description: This procedure is used alongside with the gamma/pi_inf
        !              model to transfer the density, the specific heat ratio
        !              function and liquid stiffness function from the vector
        !              of conservative or primitive variables to their scalar
        !              counterparts.
            
            
            ! Conservative or primitive variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: qK_vf
            
            ! Density and the specific heat ratio and liquid stiffness functions
            REAL(KIND(0d0)), INTENT(OUT) :: rho_K, gamma_K, pi_inf_K
            
            ! Shear and volume Reynolds numbers
            REAL(KIND(0d0)), DIMENSION(2), INTENT(OUT) :: Re_K
            
            ! Weber numbers
            REAL(KIND(0d0)), DIMENSION( num_fluids, &
                                        num_fluids  ), INTENT(OUT) :: We_K
 
            ! Partial densities and volume fractions
            REAL(KIND(0d0)), DIMENSION(num_fluids) :: alpha_rho_K, alpha_K
            
            ! Generic loop iterators
            INTEGER, INTENT(IN) :: i,j,k
            INTEGER :: l

            ! Constraining the partial densities and the volume fractions within
            ! their physical bounds to make sure that any mixture variables that
            ! are derived from them result within the limits that are set by the
            ! fluids physical parameters that make up the mixture
            ! alpha_rho_K(1) = qK_vf(i)%sf(i,j,k)
            ! alpha_K(1)     = qK_vf(E_idx+i)%sf(i,j,k)
 
            ! Performing the transfer of the density, the specific heat ratio
            ! function as well as the liquid stiffness function, respectively       
            if (model_eqns == 4) then
                rho_K    = qK_vf(1)%sf(i,j,k)
                gamma_K  = fluid_pp(1)%gamma !qK_vf(gamma_idx)%sf(i,j,k)
                pi_inf_K = fluid_pp(1)%pi_inf !qK_vf(pi_inf_idx)%sf(i,j,k)
            else if ((model_eqns == 2) .and. bubbles .and. adv_alphan) then
                rho_k = 0d0; gamma_k = 0d0; pi_inf_k = 0d0

                if (mpp_lim .and. (num_fluids > 2)) then
                    do l = 1, num_fluids
                        rho_k    = rho_k    + qK_vf(l)%sf(i,j,k) 
                        gamma_k  = gamma_k  + qK_vf(l+E_idx)%sf(i,j,k)*fluid_pp(l)%gamma
                        pi_inf_k = pi_inf_k + qK_vf(l+E_idx)%sf(i,j,k)*fluid_pp(l)%pi_inf
                    end do
                else if (num_fluids == 2) then
                    rho_K    = qK_vf(1)%sf(i,j,k)
                    gamma_K  = fluid_pp(1)%gamma
                    pi_inf_K = fluid_pp(1)%pi_inf
                else if (num_fluids > 2) then
                    do l = 1, num_fluids-1 !leave out bubble part of mixture
                        rho_k    = rho_k    + qK_vf(l)%sf(i,j,k) 
                        gamma_k  = gamma_k  + qK_vf(l+E_idx)%sf(i,j,k)*fluid_pp(l)%gamma
                        pi_inf_k = pi_inf_k + qK_vf(l+E_idx)%sf(i,j,k)*fluid_pp(l)%pi_inf
                    end do
                    !rho_K    = qK_vf(1)%sf(i,j,k)
                    !gamma_K  = fluid_pp(1)%gamma
                    !pi_inf_K = fluid_pp(1)%pi_inf
                else
                    rho_K    = qK_vf(1)%sf(i,j,k)
                    gamma_K  = fluid_pp(1)%gamma
                    pi_inf_K = fluid_pp(1)%pi_inf
                end if
            end if
            
        END SUBROUTINE s_convert_species_to_mixture_variables_bubbles ! ----------------
 
        
        
        
        SUBROUTINE s_convert_species_to_mixture_variables( qK_vf, rho_K,      &
                                                           gamma_K, pi_inf_K, &
                                                           Re_K, We_K, k,l,r  )
        ! Description: The procedure is used alongside with the volume fraction
        !              model to enable the conversion of species variables into
        !              mixture variables. Latter includes density, the specific
        !              heat ratio and liquid stiffness functions, the shear and
        !              volume Reynolds numbers and the Weber numbers.
            
            
            ! Conservative or primitive variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: qK_vf

            ! Density and the specific heat ratio and liquid stiffness functions
            REAL(KIND(0d0)), INTENT(OUT) :: rho_K, gamma_K, pi_inf_K
            
            ! Shear and volume Reynolds numbers
            REAL(KIND(0d0)), DIMENSION(2), INTENT(OUT) :: Re_K
            
            ! Weber numbers
            REAL(KIND(0d0)), DIMENSION( num_fluids, &
                                        num_fluids  ), INTENT(OUT) :: We_K
            
            ! Partial densities and volume fractions
            REAL(KIND(0d0)), DIMENSION(num_fluids) :: alpha_rho_K, alpha_K
            
            ! Generic loop iterators
            INTEGER, INTENT(IN) :: k,l,r
            
            ! Generic loop iterators
            INTEGER :: i,j
            
            ! Constraining the partial densities and the volume fractions within
            ! their physical bounds to make sure that any mixture variables that
            ! are derived from them result within the limits that are set by the
            ! fluids physical parameters that make up the mixture

            DO i = 1, num_fluids
                alpha_rho_K(i) = qK_vf(i)%sf(k,l,r)
                alpha_K(i)     = qK_vf(E_idx+i)%sf(k,l,r)
            END DO
            
            IF(mpp_lim) THEN
               
                DO i = 1, num_fluids
                    alpha_rho_K(i) = MAX(0d0,alpha_rho_K(i))
                    alpha_K(i)     = MIN(MAX(0d0,alpha_K(i)),1d0)
                END DO
               
                alpha_K = alpha_K/MAX(SUM(alpha_K),sgm_eps)
               
            END IF            
            
            ! Calculating the density, the specific heat ratio function and the
            ! liquid stiffness function, respectively, from the species analogs
            rho_K = 0d0; gamma_K = 0d0; pi_inf_K = 0d0
            
            DO i = 1, num_fluids
                rho_K    = rho_K    + alpha_rho_K(i)
                gamma_K  = gamma_K  + alpha_K(i)*fluid_pp(i)%gamma
                pi_inf_K = pi_inf_K + alpha_K(i)*fluid_pp(i)%pi_inf
            END DO
            
            
            ! Computing the shear and bulk Reynolds numbers from species analogs
            DO i = 1,2
               
                Re_K(i) = dflt_real; IF(Re_size(i) > 0) Re_K(i) = 0d0
               
                DO j = 1, Re_size(i)
                    Re_K(i) = alpha_K(Re_idx(i,j))/fluid_pp(Re_idx(i,j))%Re(i) &
                              + Re_K(i)
                END DO
               
                Re_K(i) = 1d0/MAX(Re_K(i),sgm_eps)
               
            END DO
            
            
            ! Computing the Weber numbers from species analogs
            DO i = 1, We_size
                We_K(We_idx(i,1),We_idx(i,2)) = ( alpha_K(We_idx(i,1)) + &
                                                  alpha_K(We_idx(i,2)) ) &
                             / (                  alpha_K(We_idx(i,1)) / &
                                 fluid_pp(We_idx(i,1))%We(We_idx(i,2)) + &
                                                  alpha_K(We_idx(i,2)) / &
                                 fluid_pp(We_idx(i,2))%We(We_idx(i,1)) + &
                                                               sgm_eps   )
                We_K(We_idx(i,2),We_idx(i,1)) = dflt_real
            END DO
            
        END SUBROUTINE s_convert_species_to_mixture_variables ! ----------------
        
        
        
        
        
        SUBROUTINE s_compute_roe_average_state(j,k,l) ! ------------------------
        ! Description: The goal of this subroutine is to calculate the Roe
        !              average density, velocity, enthalpy, mass fractions,
        !              specific heat ratio function and the speed of sound,
        !              at the cell-boundaries, from the left and the right
        !              cell-average variables.
            
            
            ! Generic loop iterators
            INTEGER, INTENT(IN) :: j,k,l
            
            ! Generic loop iterator
            INTEGER :: i
            
            
            rho_avg_sf(j,k,l) = SQRT(rho_L*rho_R)
            
            vel_avg = (SQRT(rho_L)*vel_L + SQRT(rho_R)*vel_R) / &
                      (SQRT(rho_L) + SQRT(rho_R))
            
            H_avg = (SQRT(rho_L)*H_L + SQRT(rho_R)*H_R) / &
                    (SQRT(rho_L) + SQRT(rho_R))
            
            DO i = 1, cont_idx%end
                mf_avg_vf(i)%sf(j,k,l) = ( SQRT(rho_L)*mf_L(i)   &
                                         + SQRT(rho_R)*mf_R(i) ) &
                                       / ( SQRT(rho_L) &
                                         + SQRT(rho_R) )
            END DO
            
            gamma_avg = (SQRT(rho_L)*gamma_L + SQRT(rho_R)*gamma_R) / &
                        (SQRT(rho_L) + SQRT(rho_R))
            
            c_avg_sf(j,k,l) = SQRT((H_avg - 5d-1*SUM(vel_avg**2d0))/gamma_avg)
            
            
        END SUBROUTINE s_compute_roe_average_state ! ---------------------------
        
        
        
        
        
        SUBROUTINE s_compute_arithmetic_average_state(j,k,l) ! -----------------
        ! Description: The goal of this subroutine is to compute the arithmetic
        !              average density, velocity, enthalpy, mass fractions, the
        !              specific heat ratio function and the sound speed, at the
        !              cell-boundaries, from the left and right cell-averages.
            
            
            ! Generic loop iterators
            INTEGER, INTENT(IN) :: j,k,l
            
            ! Generic loop iterator
            INTEGER :: i
            
            rho_avg_sf(j,k,l) = 5d-1*(rho_L + rho_R)
            vel_avg = 5d-1*(vel_L + vel_R)
            
            DO i = 1, cont_idx%end
                mf_avg_vf(i)%sf(j,k,l) = 5d-1*(mf_L(i) + mf_R(i))
            END DO
            
            H_avg = 5d-1*(H_L + H_R)
            
            gamma_avg = 5d-1*(gamma_L + gamma_R)
         
            !SHB: for Tait EOS
            alpha_avg = 5d-1*(alpha_L + alpha_R)
            pres_avg = 5d-1*(pres_L + pres_R)

            if (model_eqns .ne. 4 ) then
                c_avg_sf(j,k,l) = SQRT((H_avg - 5d-1*SUM(vel_avg**2d0))/gamma_avg)
            else !SHB: For Tait EOS
                    !fix for (1-\alpha) into sqrt
                c_avg_sf(j,k,l) = sqrt(                         &
                            (1d0/fluid_pp(1)%gamma+1d0) *       &
                            (pres_avg + fluid_pp(1)%pi_inf) /   &
                            (rho_avg_sf(j,k,l)*(1d0-alpha_avg))   &
                            )
                !from riemann solver modification
                ! c = 1/(1-alf) * sqrt( gam(p_l + \pi_inf)/\rho_l))
                ! c_L = (1/(1-alpha_L(1)))*sqrt(      &
                !            (1d0/fluid_pp(1)%gamma+1d0) *   &
                !            (pres_L + fluid_pp(1)%pi_inf)/rho_L &
                !        )
                ! c_R = (1/(1-alpha_R(1)))*sqrt(      &
                !            (1d0/fluid_pp(1)%gamma+1d0) *   &
                !            (pres_R + fluid_pp(1)%pi_inf)/rho_R &
                !        )
            end if
            
        END SUBROUTINE s_compute_arithmetic_average_state ! --------------------
        
        
        
        
        
        SUBROUTINE s_initialize_variables_conversion_module() ! ----------------
        ! Description: The computation of parameters, the allocation of memory,
        !              the association of pointers and/or the execution of any
        !              other procedures that are necessary to setup the module.
            
            
            ! Associating the procedural pointer to the appropriate subroutine
            ! that will be utilized in the conversion to the mixture variables
            
            IF (model_eqns == 1) THEN        ! gamma/pi_inf model
                s_convert_to_mixture_variables => &
                             s_convert_mixture_to_mixture_variables
            ELSEIF (bubbles) THEN        !SHB volume fraction for bubbles
                s_convert_to_mixture_variables => &
                             s_convert_species_to_mixture_variables_bubbles
            ELSE                            ! Volume fraction model
                s_convert_to_mixture_variables => &
                             s_convert_species_to_mixture_variables
            END IF
            
            
            ! Allocating velocities, mass fractions and Weber numbers as well
            ! as associating the procedural pointer to the correct subroutine
            ! utilized in the computation of the average state
            IF(char_decomp) THEN
                
                ALLOCATE(  vel_L(1:num_dims),      mf_L(1:cont_idx%end))
                ALLOCATE(  vel_R(1:num_dims),      mf_R(1:cont_idx%end))
                ALLOCATE(vel_avg(1:num_dims), mf_avg_vf(1:cont_idx%end))
                
                ALLOCATE(We_L(1:num_fluids,1:num_fluids))
                ALLOCATE(We_R(1:num_fluids,1:num_fluids))
                
                IF(avg_state == 1) THEN
                    s_compute_average_state => s_compute_roe_average_state
                ELSE
                    s_compute_average_state => s_compute_arithmetic_average_state
                END IF
                
            END IF
            
            
        END SUBROUTINE s_initialize_variables_conversion_module ! --------------
        
        
        
        
        
        SUBROUTINE s_convert_conservative_to_primitive_variables(  qK_cons_vf, &
                                                                   qK_prim_vf, &
                                                                 gm_alphaK_vf, &
                                                                     ix,iy,iz  )
        ! Description: The following procedure handles the conversion between
        !              the conservative variables and the primitive variables.
            
            
            ! Conservative variables
            TYPE(scalar_field),  &
            DIMENSION(sys_size), &
            INTENT(INOUT) :: qK_cons_vf
            
            ! Primitive variables
            TYPE(scalar_field),  &
            DIMENSION(sys_size), &
            INTENT(INOUT) :: qK_prim_vf
            
            ! Gradient magnitude of the volume fractions
            TYPE(scalar_field),        &
            ALLOCATABLE, DIMENSION(:), &
            INTENT(IN) :: gm_alphaK_vf
            
            ! Indical bounds in the x-, y- and z-directions
            TYPE(bounds_info), INTENT(IN) :: ix,iy,iz
            
            ! Density, dynamic pressure, surface energy, specific heat ratio
            ! function, liquid stiffness function, shear and volume Reynolds
            ! numbers and the Weber numbers
            REAL(KIND(0d0))                                   ::      rho_K
            REAL(KIND(0d0))                                   :: dyn_pres_K
            REAL(KIND(0d0))                                   ::     E_We_K
            REAL(KIND(0d0))                                   ::    gamma_K
            REAL(KIND(0d0))                                   ::   pi_inf_K
            REAL(KIND(0d0))                                   ::       nbub
            REAL(KIND(0d0)), DIMENSION(2)                     ::       Re_K
            REAL(KIND(0d0)), DIMENSION(num_fluids,num_fluids) ::       We_K
            REAL(KIND(0d0)), dimension(nb) :: nRtmp
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l
       
            !print*, 'cons_to_prim'
            ! Calculating the velocity and pressure from the momentum and energy
            DO l = iz%beg, iz%end
                DO k = iy%beg, iy%end
                    DO j = ix%beg, ix%end
                    
                        dyn_pres_K = 0d0; E_We_K = 0d0
                     
                        if (model_eqns .ne. 4 ) then
                            CALL s_convert_to_mixture_variables( qK_cons_vf, rho_K, &
                                                                 gamma_K, pi_inf_K, &
                                                                 Re_K, We_K, j,k,l  )
                            !no mixture variables if single bubble mixture
                        end if

                        DO i = mom_idx%beg, mom_idx%end
                            !compute velocity
                            if (model_eqns .ne. 4 ) then
                                ! u = \rho_k u / \rho_k <--- Mixture densities
                                qK_prim_vf(i)%sf(j,k,l) = qK_cons_vf(i)%sf(j,k,l) &
                                                       / MAX(rho_K,sgm_eps)
                                dyn_pres_K = dyn_pres_K + 5d-1*qK_cons_vf(i)%sf(j,k,l) &
                                                          *qK_prim_vf(i)%sf(j,k,l)
                            else !model_eqns == 4. 
                                !u = \rho u / \rho
                                !don't need mixture density here since it's the same
                                qK_prim_vf(i)%sf(j,k,l) = qK_cons_vf(i)%sf(j,k,l) &
                                                       / qK_cons_vf(1)%sf(j,k,l)
                            end if
                        END DO

                        IF (We_size > 0 .AND. (We_riemann_flux .OR. We_rhs_flux)) THEN
                             DO i = 1, We_size
                                E_We_K = E_We_K &
                                       + qK_cons_vf(E_idx+We_idx(i,1))%sf(j,k,l) * &
                                             gm_alphaK_vf(We_idx(i,2))%sf(j,k,l) / &
                                                   We_K(We_idx(i,1),We_idx(i,2))   &
                                       + qK_cons_vf(E_idx+We_idx(i,2))%sf(j,k,l) * &
                                             gm_alphaK_vf(We_idx(i,1))%sf(j,k,l) / &
                                                   We_K(We_idx(i,1),We_idx(i,2))
                             END DO
                        END IF
                    
                        ! compute pressure
                        if ( (model_eqns .ne. 4) .and. (bubbles .neqv. .TRUE.) ) then
                            ! No bubbles
                            ! p = ( E - 0.5 rho u u - pi_inf_k )/gamma_k
                            qK_prim_vf(E_idx)%sf(j,k,l) = &
                                ( qK_cons_vf(E_idx)%sf(j,k,l) &
                                                - dyn_pres_K &
                                                - pi_inf_K &
                                                - E_We_K ) / gamma_K
                        else if ( (model_eqns .ne. 4) .and. (num_fluids == 1) ) then
                            ! p_l = ( E/(1-alf) - 0.5 rho u u/(1-alf) - pi_inf_k )/gamma_k
                            qK_prim_vf(E_idx)%sf(j,k,l) = &
                                ( (qK_cons_vf(E_idx)%sf(j,k,l) &
                                                - dyn_pres_K) / (1.d0 - qK_cons_vf(alf_idx)%sf(j,k,l)) &
                                                - pi_inf_K &
                                                - E_We_K ) / gamma_K
                        else if (model_eqns .ne. 4) then
                            ! p = ( E/(1-alf) - 0.5 rho u u/(1-alf) - pi_inf_k )/gamma_k
                            qK_prim_vf(E_idx)%sf(j,k,l) = &
                                ( (qK_cons_vf(E_idx)%sf(j,k,l) &
                                                - dyn_pres_K) / (1.d0 - qK_cons_vf(alf_idx)%sf(j,k,l)) &
                                                - pi_inf_K &
                                                - E_We_K ) / gamma_K
                        else
                            ! Tait EOS
                            ! p_l = (pl0 + pi_infty)(rho/(rho_l0(1-alf)))^gamma - pi_infty  
                            qK_prim_vf(E_idx)%sf(j,k,l) =                       & 
                                   (pref+fluid_pp(1)%pi_inf) *                  &
                                   ((                                            & 
                                   qK_cons_vf(1)%sf(j,k,l)/                     &
                                   (rhoref*(1.d0-qK_cons_vf(E_idx+1)%sf(j,k,l)))   & 
                                   ) ** (1.d0/fluid_pp(1)%gamma + 1.d0)) - fluid_pp(1)%pi_inf
                        end if

                        !if (qK_prim_vf(alf_idx)%sf(j,k,l) < 0d0) qK_prim_vf(alf_idx)%sf(j,k,l) = 1.d-13
                        ! \phi = (n\phi)/n  (n = nbub)
                        if (bubbles) then
                            ! n = sqrt( 4pi/(3 alpha) * (nR)**3 )
                            do i = 1,nb
                                nRtmp(i) = qK_cons_vf(bub_idx%rs(i))%sf(j,k,l)
                                !if (nRtmp(i) < 0.d0) nRtmp(i) = 1.d-12 !stop 'nR < 0'
                            end do
                            call s_comp_n_from_cons( qK_cons_vf(alf_idx)%sf(j,k,l), nRtmp, nbub)                            
                           
                            if ( isnan(nbub) .or. (nbub < 0.d0) ) stop 'nbub is nan or negative'
                            do i = bub_idx%beg, sys_size
                                qk_prim_vf(i)%sf(j,k,l) = qk_cons_vf(i)%sf(j,k,l)/nbub
                            end do

                            !do i = 1,nb
                            !    if ( qk_prim_vf(bub_idx%rs(i))%sf(j,k,l) < 0d0 ) then
                            !        qk_prim_vf(bub_idx%rs(i))%sf(j,k,l) = 1d-12
                            !    end if
                            !end do
                        end if
                        
                    END DO
                END DO
            END DO
            

            
            
        END SUBROUTINE s_convert_conservative_to_primitive_variables ! ---------
        
        
        
        
        
        SUBROUTINE s_convert_conservative_to_characteristic_variables( & ! -----
                                   q_rs_wsK, norm_dir, is1,is2,is3, dj )
        ! Description: This procedure takes care of the conversion between the
        !              conservative variables and the characteristic variables.
            
            
            ! Conservative or characteristic cell-average variables
            TYPE(vector_field),                &
            DIMENSION(-weno_polyn:weno_polyn), &
            INTENT(INOUT) :: q_rs_wsK
            
            ! Characteristic decomposition coordinate direction
            INTEGER, INTENT(IN) :: norm_dir
            
            ! Indical bounds in the s1-, s2- and s3-directions
            TYPE(bounds_info), INTENT(IN) :: is1,is2,is3
            
            ! Evaluation location of the projection matrices
            INTEGER, INTENT(IN) :: dj
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l
            
            
            ! Initializing projection matrices for characteristic decomposition
            IF(dj == -1) CALL s_initialize_characteristic_decomposition( &
                                                               q_rs_wsK, &
                                                               norm_dir, &
                                                            is1,is2,is3  )
            
            
            ! Projecting conservative variables onto the characteristic fields
            DO i = -weno_polyn, weno_polyn
                DO l = is3%beg, is3%end
                    DO k = is2%beg, is2%end
                        DO j = is1%beg, is1%end
                        
                            IF(proc_rank == 0) THEN
                                PRINT '(A)', 'Conversion from conservative to ' // &
                                             'characteristic variables not '    // &
                                             'implemented. Exiting ...'
                                CALL s_mpi_abort()
                            END IF
                        
                        END DO
                    END DO
                END DO
            END DO
            
            
        END SUBROUTINE s_convert_conservative_to_characteristic_variables ! ----
        
        
        
        
        
        SUBROUTINE s_convert_conservative_to_flux_variables( qK_cons_vf, & ! ---
                                                                  FK_vf, &
                                                              FK_src_vf, &
                                                               ix,iy,iz  )
        ! Description: The following subroutine handles the conversion between
        !              the conservative variables and the Euler flux variables.
            
            
            ! Conservative variables
            TYPE(scalar_field),  &
            DIMENSION(sys_size), &
            INTENT(IN) :: qK_cons_vf
            
            ! Flux variables
            TYPE(scalar_field),  &
            DIMENSION(sys_size), &
            INTENT(INOUT) :: FK_vf, FK_src_vf
            
            ! Indical bounds in the x-, y- and z-directions
            TYPE(bounds_info), INTENT(IN) :: ix,iy,iz
            
            ! Generic loop iterators
            INTEGER :: j,k,l
            
            
            ! Calculating the flux variables from the conservative ones, without
            ! accounting for the contribution of either viscosity or capillarity
            DO l = iz%beg, iz%end
                DO k = iy%beg, iy%end
                    DO j = ix%beg, ix%end
                     
                        IF(proc_rank == 0) THEN
                            PRINT '(A)', 'Conversion from conservative to ' // &
                                         'flux variables not implemented. ' // &
                                         'Exiting ...'
                            CALL s_mpi_abort()
                        END IF
                     
                    END DO
                END DO
            END DO
            
            
        END SUBROUTINE s_convert_conservative_to_flux_variables ! --------------
        
        
        
        

        SUBROUTINE s_convert_primitive_to_conservative_variables(  qK_prim_vf, &
                                                                   qK_cons_vf, &
                                                                 gm_alphaK_vf, &
                                                                     ix,iy,iz  )
        ! Description: The following procedure handles the conversion between
        !              the primitive variables and the conservative variables.
            
            
            ! Primitive variables
            TYPE(scalar_field),  &
            DIMENSION(sys_size), &
            INTENT(IN) :: qK_prim_vf
            
            ! Conservative variables
            TYPE(scalar_field),  &
            DIMENSION(sys_size), &
            INTENT(INOUT) :: qK_cons_vf
            
            ! Gradient magnitude of the volume fractions
            TYPE(scalar_field),        &
            ALLOCATABLE, DIMENSION(:), &
            INTENT(IN) :: gm_alphaK_vf
            
            ! Indical bounds in the x-, y- and z-directions
            TYPE(bounds_info), INTENT(IN) :: ix,iy,iz
            
            ! Generic loop iterators
            INTEGER :: j,k,l
            
            
            ! Calculating the momentum and energy from the velocity and pressure
            DO l = iz%beg, iz%end
                DO k = iy%beg, iy%end
                    DO j = ix%beg, ix%end
                     
                        IF(proc_rank == 0) THEN
                            PRINT '(A)', 'Conversion from primitive to ' // &
                                         'conservative variables not '   // &
                                         'implemented. Exiting ...'
                            CALL s_mpi_abort()
                        END IF
                     
                    END DO
                END DO
            END DO
            
            
        END SUBROUTINE s_convert_primitive_to_conservative_variables ! ---------
        
        
        
        
        ! use this if char_decomp == .True. and weno_vars == 2 (primitive vars) !only option for char_decomp True?
        SUBROUTINE s_convert_primitive_to_characteristic_variables( & ! --------
                                q_rs_wsK, norm_dir, is1,is2,is3, dj )
        ! Description: This subroutine takes care of the conversion between the
        !              the primitive variables and the characteristic variables.
            
            
            ! Primitive or characteristic cell-average variables
            TYPE(vector_field),                &
            DIMENSION(-weno_polyn:weno_polyn), &
            INTENT(INOUT) :: q_rs_wsK
            
            ! Characteristic decomposition coordinate direction
            INTEGER, INTENT(IN) :: norm_dir
            
            ! Indical bounds in the s1-, s2- and s3-directions
            TYPE(bounds_info), INTENT(IN) :: is1,is2,is3
            
            ! Evaluation location of the projection matrices
            INTEGER, INTENT(IN) :: dj
            
            ! Cell-average partial densities, velocity and pressure
            REAL(KIND(0d0)), DIMENSION(cont_idx%end) :: alpha_rho
            REAL(KIND(0d0)), DIMENSION(num_dims)     :: vel
            REAL(KIND(0d0))                          :: pres
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l,r
            
            
            ! Initializing projection matrices for characteristic decomposition
            IF(dj == -1) CALL s_initialize_characteristic_decomposition( &
                                                               q_rs_wsK, &
                                                               norm_dir, &
                                                            is1,is2,is3  )
            
            ! I think it does this in place. take in primitive vars, set alpha rho, v, p, etc.
            ! then compute the characteristic variables

            !I think the speed of sound computation needs to be modified for Tait EOS here

            ! Projecting the primitive variables onto the characteristic fields
            DO r = -weno_polyn, weno_polyn
                DO l = is3%beg, is3%end
                    DO k = is2%beg, is2%end
                        DO j = is1%beg, is1%end
                        
                            DO i = 1, cont_idx%end
                                alpha_rho(i) = q_rs_wsK(r)%vf(i)%sf(j,k,l)
                            END DO
                        
                            DO i = 1, num_dims
                                vel(i)= q_rs_wsK(r)%vf(cont_idx%end+i)%sf(j,k,l)
                            END DO
                        
                            pres = q_rs_wsK(r)%vf(E_idx)%sf(j,k,l)
                        
                            DO i = 1, cont_idx%end
                                q_rs_wsK(r)%vf(i)%sf(j,k,l) = alpha_rho(i) &
                                        - pres * mf_avg_vf(i)%sf(j+dj,k,l) &
                                               /        c_avg_sf(j+dj,k,l) &
                                               /        c_avg_sf(j+dj,k,l)
                            END DO
                        
                            DO i = 1, num_dims
                                q_rs_wsK(r)%vf(cont_idx%end+i)%sf(j,k,l) = &
                                                           vel(dir_idx(i)) &
                                             - pres *  dir_flg(dir_idx(i)) &
                                                    / rho_avg_sf(j+dj,k,l) &
                                                    /   c_avg_sf(j+dj,k,l)
                            END DO
                        
                            q_rs_wsK(r)%vf(E_idx)%sf(j,k,l) = vel(dir_idx(1)) &
                                                + pres / rho_avg_sf(j+dj,k,l) &
                                                       /   c_avg_sf(j+dj,k,l)
                        
                        END DO
                    END DO
                END DO
            END DO
            
            
        END SUBROUTINE s_convert_primitive_to_characteristic_variables ! -------
        
        
        
        
        ! this is never used. the fluxes are computed in the riemann solver subroutine 
        SUBROUTINE s_convert_primitive_to_flux_variables( qK_prim_vf, & ! ------
                                                               FK_vf, &
                                                           FK_src_vf, &
                                                            ix,iy,iz  )
        ! Description: The following subroutine handles the conversion between
        !              the primitive variables and the Eulerian flux variables.
            
            
            ! Primitive variables
            TYPE(scalar_field),  &
            DIMENSION(sys_size), &
            INTENT(IN) :: qK_prim_vf
            
            ! Flux variables
            TYPE(scalar_field),  &
            DIMENSION(sys_size), &
            INTENT(INOUT) :: FK_vf, FK_src_vf
            
            ! Indical bounds in the x-, y- and z-directions
            TYPE(bounds_info), INTENT(IN) :: ix,iy,iz
            
            ! Partial densities, density, velocity, pressure, energy, advection
            ! variables, the specific heat ratio and liquid stiffness functions,
            ! the shear and volume Reynolds numbers and the Weber numbers
            REAL(KIND(0d0)), DIMENSION(cont_idx%end)          :: alpha_rho_K
            REAL(KIND(0d0))                                   ::       rho_K
            REAL(KIND(0d0)), DIMENSION(num_dims)              ::       vel_K
            REAL(KIND(0d0))                                   ::      pres_K
            REAL(KIND(0d0))                                   ::         E_K
            REAL(KIND(0d0)), DIMENSION(adv_idx%end-E_idx)     ::       adv_K
            REAL(KIND(0d0))                                   ::     gamma_K
            REAL(KIND(0d0))                                   ::    pi_inf_K
            REAL(KIND(0d0)), DIMENSION(2)                     ::        Re_K
            REAL(KIND(0d0)), DIMENSION(num_fluids,num_fluids) ::        We_K
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l
            
            
            ! Computing the flux variables from the primitive variables, without
            ! accounting for the contribution of either viscosity or capillarity
            DO l = iz%beg, iz%end
                DO k = iy%beg, iy%end
                    DO j = ix%beg, ix%end
                     
                        DO i = 1, cont_idx%end
                            alpha_rho_K(i) = qK_prim_vf(i)%sf(j,k,l)
                        END DO
                     
                        DO i = 1, num_dims
                            vel_K(i) = qK_prim_vf(cont_idx%end+i)%sf(j,k,l)
                        END DO
                     
                        pres_K = qK_prim_vf(E_idx)%sf(j,k,l)
                     
                        CALL s_convert_to_mixture_variables( qK_prim_vf, rho_K, &
                                                             gamma_K, pi_inf_K, &
                                                             Re_K, We_K, j,k,l  )

                        ! Computing the energy from the pressure
                        !if ( (model_eqns .ne. 4) .and. bubbles .neqv. .TRUE. ) then
                            ! E = Gamma*P + \rho u u /2 + \pi_inf
                            E_K = gamma_K*pres_K + pi_inf_K &
                                 + 5d-1*rho_K*SUM(vel_K**2d0)
                        !else if ( model_eqns .ne. 4 .and. bubbles ) then
                        !    ! \tilde{E} = dyn_pres + (1-\alf)(\Gamma p_l + \Pi_inf)
                        !    E_K = (1.d0 - qK_prim_vf(alf_idx)%sf(j,k,l)) * (gamma_K*pres_K + pi_inf_K) &
                        !        + 5d-1*rho_K*SUM(vel_K**2d0)
                        !else
                        !    !Tait EOS, no conserved energy variable
                        !    E_K = 0.d0
                        !end if

                   
                         ! this is just \alpha_1 (for one fluid)
                        DO i = 1, adv_idx%end-E_idx
                            adv_K(i) = qK_prim_vf(E_idx+i)%sf(j,k,l)
                        END DO
                     
                        ! mass flux, this should be \alpha_i \rho_i u_i 
                        ! only x-dir? vel_k(dir_idx(1))
                        DO i = 1, cont_idx%end
                            FK_vf(i)%sf(j,k,l) = alpha_rho_K(i)*vel_K(dir_idx(1))
                        END DO
                     
                        ! momentum flux in num_dims directions, this is \rho u u + p I
                        DO i = 1, num_dims
                            FK_vf(cont_idx%end+dir_idx(i))%sf(j,k,l) = &
                                               rho_K*vel_K(dir_idx(1)) &
                                                    *vel_K(dir_idx(i)) &
                                          + pres_K*dir_flg(dir_idx(i))
                        END DO
                    
                        ! energy flux, u(E+p)
                        FK_vf(E_idx)%sf(j,k,l) = vel_K(dir_idx(1))*(E_K + pres_K)
                     
                        ! have been using == 2 
                        IF(riemann_solver == 1) THEN
                        
                            DO i = adv_idx%beg, adv_idx%end
                                FK_vf(i)%sf(j,k,l) = 0d0
                                FK_src_vf(i)%sf(j,k,l) = adv_K(i-E_idx)
                            END DO
                        
                        ELSE
                            ! u*\alpha_1
                            DO i = adv_idx%beg, adv_idx%end
                                FK_vf(i)%sf(j,k,l) = vel_K(dir_idx(1))*adv_K(i-E_idx)
                            END DO
                        
                            ! for \alpha_1 \div u
                            FK_src_vf(adv_idx%beg)%sf(j,k,l) = vel_K(dir_idx(1))
                        END IF
                    
                        !if (bubbles) then
                        !    do i = 1,Nb
                        !        FK_vf(bub_idx%rs(i))%sf(j,k,l) = vel_K(dir_idx(1))*qK_prim_vf(bub_idx%rs(i))%sf(j,k,l)
                        !        FK_vf(bub_idx%vs(i))%sf(j,k,l) = vel_K(dir_idx(1))*qK_prim_vf(bub_idx%vs(i))%sf(j,k,l)
                        !    end do
                        !end if
                    END DO
                END DO
            END DO
            
            
        END SUBROUTINE s_convert_primitive_to_flux_variables ! -----------------
        
        
        SUBROUTINE s_convert_primitive_to_flux_variables_bubbles( qK_prim_vf, & ! ------
                                                                qk_cons_vf, &
                                                                FK_vf, &
                                                                FK_src_vf, &
                                                                ix,iy,iz  )
        ! Description: The following subroutine handles the conversion between
        !              the primitive variables and the Eulerian flux variables.
            
            
            ! Primitive variables
            TYPE(scalar_field),  &
            DIMENSION(sys_size), &
            INTENT(IN) :: qK_prim_vf, qK_cons_vf
            
            ! Flux variables
            TYPE(scalar_field),  &
            DIMENSION(sys_size), &
            INTENT(INOUT) :: FK_vf, FK_src_vf
            
            ! Indical bounds in the x-, y- and z-directions
            TYPE(bounds_info), INTENT(IN) :: ix,iy,iz
            
            ! Partial densities, density, velocity, pressure, energy, advection
            ! variables, the specific heat ratio and liquid stiffness functions,
            ! the shear and volume Reynolds numbers and the Weber numbers
            REAL(KIND(0d0)), DIMENSION(cont_idx%end)          :: alpha_rho_K
            REAL(KIND(0d0))                                   ::       rho_K
            REAL(KIND(0d0)), DIMENSION(num_dims)              ::       vel_K
            REAL(KIND(0d0))                                   ::      pres_K
            REAL(KIND(0d0))                                   ::         E_K
            REAL(KIND(0d0)), DIMENSION(adv_idx%end-E_idx)     ::       adv_K
            REAL(KIND(0d0))                                   ::     gamma_K
            REAL(KIND(0d0))                                   ::    pi_inf_K
            REAL(KIND(0d0)), DIMENSION(2)                     ::        Re_K
            REAL(KIND(0d0)), DIMENSION(num_fluids,num_fluids) ::        We_K
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l
            
            dir_idx = (/1,2,3/); dir_flg = (/1d0,0d0,0d0/)
 
            ! Computing the flux variables from the primitive variables, without
            ! accounting for the contribution of either viscosity or capillarity
            DO l = iz%beg, iz%end
                DO k = iy%beg, iy%end
                    DO j = ix%beg, ix%end
                     
                        DO i = 1, cont_idx%end
                            alpha_rho_K(i) = qK_prim_vf(i)%sf(j,k,l)
                        END DO
                     
                        DO i = 1, num_dims
                            vel_K(i) = qK_prim_vf(cont_idx%end+i)%sf(j,k,l)
                        END DO
                     
                        pres_K = qK_prim_vf(E_idx)%sf(j,k,l)
                     
                        CALL s_convert_to_mixture_variables( qK_prim_vf, rho_K, &
                                                             gamma_K, pi_inf_K, &
                                                             Re_K, We_K, j,k,l  )
                     
                        ! mass flux, \rho u
                        DO i = 1, cont_idx%end
                            FK_vf(i)%sf(j,k,l) = alpha_rho_K(i)*vel_K(dir_idx(1))
                        END DO
                     
                        ! momentum flux, \rho u u + p I
                        DO i = 1, num_dims
                            FK_vf(cont_idx%end+dir_idx(i))%sf(j,k,l) = &
                                               rho_K*vel_K(dir_idx(1)) &
                                                    *vel_K(dir_idx(i)) &
                                          + pres_K*dir_flg(dir_idx(i))
                        END DO
                    
                        ! energy flux, 0
                        FK_vf(E_idx)%sf(j,k,l) = 0. 

                        ! vol. frac, nR, and nRdot fluxes, u{\alpha, nR, nRdot}
                        DO i = adv_idx%beg, sys_size
                            FK_vf(i)%sf(j,k,l) = vel_K(dir_idx(1))*qK_cons_vf(i)%sf(j,k,l)
                        END DO

                        
                        !print*, vel_K(dir_idx(1)), qK_prim_vf(2)%sf(j,k,l), FK_vf(adv_idx%beg)%sf(j,k,l)
                    END DO
                END DO
            END DO
           
            !stop
            
        END SUBROUTINE s_convert_primitive_to_flux_variables_bubbles ! -----------------        
        
        
        SUBROUTINE s_convert_characteristic_to_conservative_variables( & ! -----
                                             qK_rs_vf, is1,is2,is3, dj )
        ! Description: This procedure takes care of the conversion between the
        !              characteristic variables and the conservative variables.
            
            
            ! Characteristic or conservative cell-boundary variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: qK_rs_vf
            
            ! Indical bounds in the s1-, s2- and s3-directions
            TYPE(bounds_info), INTENT(IN) :: is1,is2,is3
            
            ! Evaluation location of the projection matrices
            INTEGER, INTENT(IN) :: dj
            
            ! Generic loop iterators
            INTEGER :: j,k,l
            
            
            ! Projecting characteristic variables onto the conservative fields
            DO l = is3%beg, is3%end
                DO k = is2%beg, is2%end
                    DO j = is1%beg, is1%end
                     
                        IF(proc_rank == 0) THEN
                            PRINT '(A)', 'Conversion from characteristic to ' // &
                                         'conservative variables not '        // &
                                         'implemented. Exiting ...'
                            CALL s_mpi_abort()
                        END IF
                     
                    END DO
                END DO
            END DO
            
            
            ! Finalizing projection matrices for characteristic decomposition
            IF(dj == 0) CALL s_finalize_characteristic_decomposition()
            
            
        END SUBROUTINE s_convert_characteristic_to_conservative_variables ! ----
        
        
        
        
        
        SUBROUTINE s_convert_characteristic_to_primitive_variables( & ! --------
                                          qK_rs_vf, is1,is2,is3, dj )
        ! Description: This subroutine takes care of the conversion between the
        !              the characteristic variables and the primitive variables.
            
            
            ! Characteristic or primitive cell-boundary variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: qK_rs_vf
            
            ! Indical bounds in the s1-, s2- and s3-directions
            TYPE(bounds_info), INTENT(IN) :: is1,is2,is3
            
            ! Evaluation location of the projection matrices
            INTEGER, INTENT(IN) :: dj
            
            ! Characteristic cell-boundary variables
            REAL(KIND(0d0)), DIMENSION(E_idx) :: w_K
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l
            
            
            ! Projecting characteristic variables onto the primitive fields
            DO l = is3%beg, is3%end
                DO k = is2%beg, is2%end
                    DO j = is1%beg, is1%end
                    
                        DO i = 1, E_idx
                            w_K(i) = qK_rs_vf(i)%sf(j,k,l)
                        END DO
                    
                        DO i = 1, cont_idx%end
                            qK_rs_vf(i)%sf(j,k,l) = w_K(i) + rho_avg_sf(j+dj,k,l) * &
                                                        mf_avg_vf(i)%sf(j+dj,k,l) / &
                                                               c_avg_sf(j+dj,k,l) * &
                                                          5d-1*( w_K(   E_idx   )   &
                                                             - w_K(mom_idx%beg) )
                        END DO
                    
                        DO i = 1, num_dims
                            qK_rs_vf(cont_idx%end+dir_idx(i))%sf(j,k,l) = &
                                                ( w_K(cont_idx%end+i)     &
                                                + w_K(     E_idx    ) *   &
                                                  dir_flg(dir_idx(i)) ) / &
                                          ( 1d0 + dir_flg(dir_idx(i)) )
                        END DO
                    
                        qK_rs_vf(E_idx)%sf(j,k,l) = 5d-1*rho_avg_sf(j+dj,k,l) * &
                                                           c_avg_sf(j+dj,k,l) * &
                                                           ( w_K(   E_idx   )   &
                                                           - w_K(mom_idx%beg) )
                    
                    END DO
                END DO
            END DO
            
            
            ! Finalizing projection matrices for characteristic decomposition
            IF(dj == 0) CALL s_finalize_characteristic_decomposition()
            
            
        END SUBROUTINE s_convert_characteristic_to_primitive_variables ! -------
        
        
        
        
        
        SUBROUTINE s_initialize_characteristic_decomposition( & ! --------------
                              q_rs_wsK, norm_dir, is1,is2,is3 )
        ! Description: The computation of parameters, the allocation of memory,
        !              the association of pointers and/or the execution of any
        !              other procedures needed to configure the characteristic
        !              decomposition.
            
            
            ! Conservative or primitive cell-average variables
            TYPE(vector_field),                &
            DIMENSION(-weno_polyn:weno_polyn), &
            INTENT(IN) :: q_rs_wsK
            
            ! Characteristic decomposition coordinate direction
            INTEGER, INTENT(IN) :: norm_dir
            
            ! Indical bounds in the s1-, s2- and s3-directions
            TYPE(bounds_info), INTENT(IN) :: is1,is2,is3
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l,r
            
            
            ! Configuring the coordinate direction indexes and flags
            IF(norm_dir == 1) THEN
                dir_idx = (/1,2,3/); dir_flg = (/1d0,0d0,0d0/)
            ELSEIF(norm_dir == 2) THEN
                dir_idx = (/2,1,3/); dir_flg = (/0d0,1d0,0d0/)
            ELSE
                dir_idx = (/3,1,2/); dir_flg = (/0d0,0d0,1d0/)
            END IF
            
            
            ! Allocating the average density, speed of sound and mass fractions
            ALLOCATE(rho_avg_sf( is1%beg-1 : is1%end, &
                                 is2%beg   : is2%end, &
                                 is3%beg   : is3%end ))
            ALLOCATE(  c_avg_sf( is1%beg-1 : is1%end, &
                                 is2%beg   : is2%end, &
                                 is3%beg   : is3%end ))
            
            DO i = 1, cont_idx%end
                ALLOCATE(mf_avg_vf(i)%sf( is1%beg-1 : is1%end, &
                                          is2%beg   : is2%end, &
                                          is3%beg   : is3%end ))
            END DO
            
            
            ! Performing evaluation of projection matrices at cell-boundaries
            DO r = -1,0
                DO l = is3%beg, is3%end
                    DO k = is2%beg, is2%end
                        DO j = is1%beg, is1%beg + (r+1)*(is1%end - is1%beg)
                        
                            CALL s_convert_to_mixture_variables( q_rs_wsK( r )%vf, &
                                                                   rho_L, gamma_L, &
                                                                   pi_inf_L, Re_L, &
                                                                      We_L, j,k,l  )
                            CALL s_convert_to_mixture_variables( q_rs_wsK(r+1)%vf, &
                                                                   rho_R, gamma_R, &
                                                                   pi_inf_R, Re_R, &
                                                                      We_R, j,k,l  )
                            
                            DO i = 1, num_dims
                                vel_L(i) = q_rs_wsK( r )%vf(cont_idx%end+i)%sf(j,k,l)
                                vel_R(i) = q_rs_wsK(r+1)%vf(cont_idx%end+i)%sf(j,k,l)
                            END DO

                            !SHB alpha_L/R for mixture SOS computation
                            alpha_L = q_rs_wsK(r)%vf(E_idx+1)%sf(j,k,l)
                            alpha_R = q_rs_wsK(r+1)%vf(E_idx+1)%sf(j,k,l)

                            IF(weno_vars == 1) THEN
                               
                                E_L = q_rs_wsK( r )%vf(E_idx)%sf(j,k,l)
                                E_R = q_rs_wsK(r+1)%vf(E_idx)%sf(j,k,l)
                               
                                pres_L = - ( 5d-1*rho_L*SUM(vel_L**2d0) &
                                           + pi_inf_L - E_L ) / gamma_L
                                pres_R = - ( 5d-1*rho_R*SUM(vel_R**2d0) &
                                           + pi_inf_R - E_R ) / gamma_R
                               
                            ELSE
                               
                                pres_L = q_rs_wsK( r )%vf(E_idx)%sf(j,k,l)
                                pres_R = q_rs_wsK(r+1)%vf(E_idx)%sf(j,k,l)
                               
                                E_L = gamma_L*pres_L + pi_inf_L &
                                    + 5d-1*rho_L*SUM(vel_L**2d0)
                                E_R = gamma_R*pres_R + pi_inf_R &
                                    + 5d-1*rho_R*SUM(vel_R**2d0)
                               
                            END IF
                        
                            H_L = (E_L + pres_L)/rho_L
                            H_R = (E_R + pres_R)/rho_R

                            DO i = 1, cont_idx%end
                                mf_L(i) = q_rs_wsK( r )%vf(i)%sf(j,k,l)/rho_L
                                mf_R(i) = q_rs_wsK(r+1)%vf(i)%sf(j,k,l)/rho_R
                            END DO
                        
                            CALL s_compute_average_state(j+r,k,l)
                        
                        END DO
                    END DO
                END DO
            END DO
            
            
        END SUBROUTINE s_initialize_characteristic_decomposition ! -------------
        
        
        
        
        
        SUBROUTINE s_finalize_characteristic_decomposition() ! -----------------
        ! Description: Deallocation and/or disassociation procedures that are
        !              necessary to finalize the characteristic decomposition
            
            
            ! Generic loop iterator
            INTEGER :: i
            
            
            ! Deallocating the average density, sound speed and mass fractions
            DEALLOCATE(rho_avg_sf, c_avg_sf)
            
            DO i = 1, cont_idx%end
                DEALLOCATE(mf_avg_vf(i)%sf)
            END DO
            
            
        END SUBROUTINE s_finalize_characteristic_decomposition ! ---------------
        
        
        
        
        
        SUBROUTINE s_compute_lsq_gradient_curvature(v_vf,grad_x_vf,grad_y_vf,grad_z_vf,norm_vf,kappa_vf)

            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN)      :: v_vf
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT)   :: grad_x_vf, grad_y_vf, grad_z_vf
            TYPE(scalar_field), DIMENSION(num_fluids), INTENT(INOUT) :: norm_vf
            TYPE(scalar_field), DIMENSION(num_fluids), INTENT(INOUT) :: kappa_vf
            REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:,:) :: A
            REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:) :: sol, b, AA
            REAL(KIND(0d0)) :: xloc, yloc, zloc
            INTEGER :: ndim
            INTEGER :: i,j,k, ii, jj, kk, i1, i2, l
            INTEGER :: stencil_i_min, stencil_i_max
            INTEGER :: stencil_j_min, stencil_j_max
            INTEGER :: stencil_k_min, stencil_k_max
            TYPE(bounds_info) :: ix,iy,iz
            REAL(KIND(0d0)) :: norm, kappa

            IF (p > 0) THEN ! 3D simulation
                ALLOCATE(A(10,10))
                ALLOCATE(sol(10)); ALLOCATE(b(10)); ALLOCATE(AA(10))
            ELSE ! 2D simulation
                ALLOCATE(A(6,6))
                ALLOCATE(sol(6)); ALLOCATE(b(6)); ALLOCATE(AA(6))
            END IF

            ! Parabolic fitting
            ndim = SIZE(sol,1)

            ix%beg = -buff_size; iy%beg = -buff_size
            ix%end = m + buff_size; iy%end = n + buff_size
            IF (p > 0) THEN
                iz%beg = -buff_size; iz%end = p + buff_size
            ELSE
                iz%beg = -1; iz%end = 1
            END IF

            DO l = 1, crv_size
                DO k = iz%beg+1, iz%end-1
                    DO j = iy%beg+1, iy%end-1
                        DO i = ix%beg+1, ix%end-1
                            A(:,:) = 0d0
                            b(:) = 0d0
                            sol(:) = 0d0
                            AA(:) = 0d0
    
                            stencil_i_min = i-1; stencil_i_max = i+1
                            stencil_j_min = j-1; stencil_j_max = j+1
                            IF (p > 0) THEN
                                stencil_k_min = k-1; stencil_k_max = k+1
                            ELSE 
                                stencil_k_min = 0; stencil_k_max = 0
                            END IF
    
                            DO kk = stencil_k_min, stencil_k_max
                                DO jj = stencil_j_min, stencil_j_max
                                    DO ii = stencil_i_min, stencil_i_max
    
                                        ! Ignore corner points in 3D stencil
                                        IF ((p > 0).AND.(ABS(ii-i)==1).AND.(ABS(jj-j)==1).AND.(ABS(kk-k)==1)) CYCLE
    
                                        ! Find distance between cell centers
                                        xloc = x_cc(ii) - x_cc(i)
                                        yloc = y_cc(jj) - y_cc(j)
                                        IF (p > 0) zloc = z_cc(kk) - z_cc(k)
    
                                        ! Compute operator
                                        AA(1) = 1d0
                                        AA(2) = xloc
                                        AA(3) = yloc
                                        AA(4) = 5d-1*xloc**2d0
                                        AA(5) = 5d-1*yloc**2d0
                                        AA(6) = xloc*yloc
                                        IF (p > 0) THEN
                                            AA(7) = zloc
                                            AA(8) = 5d-1*zloc**2d0
                                            AA(9) = yloc*zloc
                                            AA(10) = zloc*xloc
                                        END IF
    
                                        DO i1 = 1, ndim
                                            DO i2 = 1, ndim
                                                A(i1,i2) = A(i1,i2) + AA(i1)*AA(i2)
                                            END DO
                                        END DO
    
                                        ! Form RHS vector
                                        DO i1 = 1, ndim
                                            b(i1) = b(i1) + v_vf(E_idx+crv_idx(l))%sf(ii,jj,kk)*AA(i1)
                                        END DO
                                    END DO
                                END DO
                            END DO
    
                            CALL s_solve_linear_system(A,b,sol,ndim)
    
                            IF (p > 0) THEN
                                norm = SQRT(sol(2)**2d0 + sol(3)**2d0 + sol(7)**2d0)
                            ELSE
                                norm = SQRT(sol(2)**2d0 + sol(3)**2d0)
                            END IF
                            
                            grad_x_vf(E_idx+crv_idx(l))%sf(i,j,k) = sol(2)
                            grad_y_vf(E_idx+crv_idx(l))%sf(i,j,k) = sol(3)
                            IF (p > 0) THEN
                                grad_z_vf(E_idx+crv_idx(l))%sf(i,j,k) = sol(7)
                            END IF
                           
                            IF (p > 0) THEN
                                kappa = -(sol(2)**2d0*sol(5) - 2d0*sol(2)*sol(3)*sol(6) + sol(2)**2d0*sol(8) + &
                                          sol(3)**2d0*sol(8) - 2d0*sol(3)*sol(7)*sol(9) + sol(3)**2d0*sol(4) + &
                                          sol(7)**2d0*sol(4) - 2d0*sol(7)*sol(2)*sol(10)+ sol(7)**2d0*sol(5))/MAX(norm,sgm_eps)**3d0
                            ELSE
                                kappa = -(sol(2)**2d0*sol(5) - 2d0*sol(2)*sol(3)*sol(6) + sol(3)**2d0*sol(4))/MAX(norm,sgm_eps)**3d0
                            END IF
                            norm_vf(crv_idx(l))%sf(i,j,k) = norm
                            kappa_vf(crv_idx(l))%sf(i,j,k) = kappa
                        END DO
                    END DO
                END DO
            END DO
            
            DEALLOCATE(A,sol,b,AA)

        END SUBROUTINE s_compute_lsq_gradient_curvature





        SUBROUTINE s_solve_linear_system(A,b,sol,ndim)

            INTEGER, INTENT(IN) :: ndim
            REAL(KIND(0d0)), DIMENSION(ndim,ndim), INTENT(INOUT) :: A
            REAL(KIND(0d0)), DIMENSION(ndim), INTENT(INOUT) :: b
            REAL(KIND(0d0)), DIMENSION(ndim), INTENT(OUT) :: sol

!            INTEGER, DIMENSION(ndim) :: ipiv
!            INTEGER :: nrhs, lda, ldb, info
!            EXTERNAL DGESV

            INTEGER :: i,j,k


            ! Solve linear system using Intel MKL (Hooke)
!            nrhs = 1
!            lda = ndim
!            ldb = ndim
! 
!            CALL DGESV(ndim, nrhs, A, lda, ipiv, b, ldb, info)
! 
!            DO i = 1, ndim
!                sol(i) = b(i)
!            END DO
! 
!            IF (info /= 0) THEN
!                PRINT '(A)', 'Trouble solving linear system'
!                CALL s_mpi_abort()
!            END IF

            ! Solve linear system using own linear solver (Thomson/Darter/Comet/Stampede)
            ! Forward elimination
            DO i = 1, ndim
                ! Pivoting
                j = i-1+MAXLOC(ABS(A(i:ndim,i)),1)
                sol = A(i,:)
                A(i,:) = A(j,:)
                A(j,:) = sol
                sol(1) = b(i)
                b(i) = b(j)
                b(j) = sol(1)
                ! Elimination
                b(i) = b(i) / A(i,i)
                A(i,:) = A(i,:) / A(i,i)
                DO k = i+1, ndim
                    b(k) = b(k) - A(k,i)*b(i)
                    A(k,:) = A(k,:) - A(k,i)*A(i,:)
                END DO
            END DO

            ! Backward substitution
            DO i = ndim, 1, -1
                sol(i) = b(i)
                DO k = i+1, ndim
                    sol(i) = sol(i) - A(i,k)*sol(k)
                END DO
            END DO

        END SUBROUTINE s_solve_linear_system    
        
        
        
        
        
        SUBROUTINE s_finalize_variables_conversion_module() ! ------------------
        ! Description: Module deallocation and/or disassociation procedures
            
            
            ! Disassociating the pointer to the procedure that was utilized to
            ! to convert mixture or species variables to the mixture variables
            s_convert_to_mixture_variables => NULL()
            
            
            ! Deallocating the velocities, the mass fractions, and the Weber
            ! numbers as well as disassociating the pointer to the procedure
            ! which was used in the calculation of the average state
            IF(char_decomp) THEN
                
                DEALLOCATE(vel_L  , mf_L     )
                DEALLOCATE(vel_R  , mf_R     )
                DEALLOCATE(vel_avg, mf_avg_vf)
                
                DEALLOCATE(We_L, We_R)
                
                s_compute_average_state => NULL()
                
            END IF
            
            
        END SUBROUTINE s_finalize_variables_conversion_module ! ----------------
        
        
        
        
        
END MODULE m_variables_conversion