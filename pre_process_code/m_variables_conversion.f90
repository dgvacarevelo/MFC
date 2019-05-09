! MFC v3.0 - Pre-process Code: m_variables_conversion.f90
! Description: This module consists of subroutines used in the conversion of the
!              conservative variables into the primitive ones and vice versa. In
!              addition, the module also contains the subroutines used to obtain
!              the mixture variables.
! Author: Vedran Coralic
! Date: 06/10/12


MODULE m_variables_conversion
    
    
    ! Dependencies =============================================================
    USE m_derived_types         ! Definitions of the derived types
    
    USE m_global_parameters     ! Global parameters for the code
    ! ==========================================================================
    
    
    IMPLICIT NONE
    
    
    ! Abstract interface to two subroutines designed for the transfer/conversion
    ! of the mixture/species variables to the mixture variables
    ABSTRACT INTERFACE
        
        SUBROUTINE s_convert_xxxxx_to_mixture_variables(   q_vf, i,j,k,   &
                                                         rho,gamma,pi_inf )
        ! Description: Structure of the s_convert_mixture_to_mixture_variables
        !              and s_convert_species_to_mixture_variables subroutines
            
            ! Importing the derived type scalar_field from m_derived_types.f90
            ! and global variable sys_size, from m_global_variables.f90, as
            ! the abstract interface does not inherently have access to them
            IMPORT :: scalar_field, sys_size
            
            ! Conservative or primitive variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: q_vf
            
            ! Cell indexes for which to transfer/compute mixture variables
            INTEGER, INTENT(IN) :: i,j,k
            
            ! Density and the specific heat ratio and liquid stiffness functions
            REAL(KIND(0d0)), INTENT(OUT) :: rho
            REAL(KIND(0d0)), INTENT(OUT) :: gamma
            REAL(KIND(0d0)), INTENT(OUT) :: pi_inf
            
        END SUBROUTINE s_convert_xxxxx_to_mixture_variables
        
    END INTERFACE
    
    ! NOTE: These abstract interfaces allow for the declaration of a pointer to
    ! a procedure such that the choice of the model equations does not have to
    ! be queried every time the mixture quantities are needed.
    
    
    ! Pointer referencing the subroutine s_convert_mixture_to_mixture_variables
    ! or s_convert_species_to_mixture_variables, based on model equations choice
    PROCEDURE(s_convert_xxxxx_to_mixture_variables), &
    POINTER :: s_convert_to_mixture_variables => NULL()
    
    
    CONTAINS
        
        SUBROUTINE s_convert_mixture_to_mixture_variables(     q_vf, i,j,k,    &
                                                             rho,gamma,pi_inf  )
        ! Description: This subroutine is designed for the gamma/pi_inf model
        !              and provided a set of either conservative or primitive
        !              variables, transfers the density, specific heat ratio
        !              function and the liquid stiffness function from q_vf to
        !              rho, gamma and pi_inf.
            
            
            ! Conservative or primitive variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: q_vf
            
            ! Indexes of the cell for which to transfer the mixture variables
            INTEGER, INTENT(IN) :: i,j,k
            
            ! Density, the specific heat ratio function and liquid stiffness
            ! function, respectively
            REAL(KIND(0d0)), INTENT(OUT) :: rho
            REAL(KIND(0d0)), INTENT(OUT) :: gamma
            REAL(KIND(0d0)), INTENT(OUT) :: pi_inf
            
            
            ! Transfering the density, the specific heat ratio function and the
            ! liquid stiffness function, respectively
            rho    = q_vf(1)%sf(i,j,k)
            gamma  = q_vf(gamma_idx)%sf(i,j,k)
            pi_inf = q_vf(pi_inf_idx)%sf(i,j,k)
            
            
        END SUBROUTINE s_convert_mixture_to_mixture_variables ! ----------------
        
        SUBROUTINE s_convert_species_to_mixture_variables_bubbles ( qK_vf,  &
                                                            j,k,l,&
                                                            rho_K,gamma_K, pi_inf_K &
                                                             )

        ! Description: This procedure is used alongside with the gamma/pi_inf
        !              model to transfer the density, the specific heat ratio
        !              function and liquid stiffness function from the vector
        !              of conservative or primitive variables to their scalar
        !              counterparts.
            
            
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: qK_vf !should be primitive variables
            
            ! Density and the specific heat ratio and liquid stiffness functions
            REAL(KIND(0d0)), INTENT(OUT) :: rho_K, gamma_K, pi_inf_K
            
            
            ! Partial densities and volume fractions
            REAL(KIND(0d0)), DIMENSION(num_fluids) :: alpha_rho_K, alpha_K
            
            ! Generic loop iterators
            INTEGER, INTENT(IN) :: j,k,l
            INTEGER :: i
            
            ! Constraining the partial densities and the volume fractions within
            ! their physical bounds to make sure that any mixture variables that
            ! are derived from them result within the limits that are set by the
            ! fluids physical parameters that make up the mixture
            ! alpha_rho_K(1) = qK_vf(i)%sf(i,j,k)
            ! alpha_K(1)     = qK_vf(E_idx+i)%sf(i,j,k)
 
            ! Performing the transfer of the density, the specific heat ratio
            ! function as well as the liquid stiffness function, respectively 

            if (model_eqns == 4) then
                rho_K    = qK_vf(1)%sf(j,k,l)
                gamma_K  = fluid_pp(1)%gamma    !qK_vf(gamma_idx)%sf(i,j,k)
                pi_inf_K = fluid_pp(1)%pi_inf   !qK_vf(pi_inf_idx)%sf(i,j,k)       
            else if ((model_eqns == 2) .and. bubbles .and. adv_alphan) then
                rho_k = 0d0; gamma_k = 0d0; pi_inf_k = 0d0

                if (mpp_lim .and. (num_fluids > 2)) then
                    do i = 1,num_fluids
                        rho_k    = rho_k    + qK_vf(i)%sf(j,k,l) 
                        gamma_k  = gamma_k  + qK_vf(i+E_idx)%sf(j,k,l)*fluid_pp(i)%gamma
                        pi_inf_k = pi_inf_k + qK_vf(i+E_idx)%sf(j,k,l)*fluid_pp(i)%pi_inf
                    end do
                else if (num_fluids == 2) then
                    rho_K    = qK_vf(1)%sf(j,k,l)
                    gamma_K  = fluid_pp(1)%gamma
                    pi_inf_K = fluid_pp(1)%pi_inf   
                else if (num_fluids > 2) then
                    do i = 1, num_fluids-1 !leave out bubble part of mixture
                        rho_k    = rho_k    + qK_vf(i)%sf(j,k,l) 
                        gamma_k  = gamma_k  + qK_vf(i+E_idx)%sf(j,k,l)*fluid_pp(i)%gamma
                        pi_inf_k = pi_inf_k + qK_vf(i+E_idx)%sf(j,k,l)*fluid_pp(i)%pi_inf
                    end do
                    !rho_K    = qK_vf(1)%sf(j,k,l)
                    !gamma_K  = fluid_pp(1)%gamma
                    !pi_inf_K = fluid_pp(1)%pi_inf   
                else
                    rho_K    = qK_vf(1)%sf(j,k,l)
                    gamma_K  = fluid_pp(1)%gamma
                    pi_inf_K = fluid_pp(1)%pi_inf   
                end if
            end if

        END SUBROUTINE s_convert_species_to_mixture_variables_bubbles ! ----------------
        
        SUBROUTINE s_convert_species_to_mixture_variables(     q_vf, j,k,l,    &
                                                             rho,gamma,pi_inf  )
        ! Description: This subroutine is designed for the volume fraction model
        !              and provided a set of either conservative or primitive
        !              variables, computes the density, the specific heat ratio
        !              function and the liquid stiffness function from q_vf and
        !              stores the results into rho, gamma and pi_inf.
            
            
            ! Conservative or primitive variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: q_vf
            
            ! Indexes of the cell for which to compute the mixture variables
            INTEGER, INTENT(IN) :: j,k,l
            
            ! Density, the specific heat ratio function and the liquid stiffness
            ! function, respectively
            REAL(KIND(0d0)), INTENT(OUT) :: rho
            REAL(KIND(0d0)), INTENT(OUT) :: gamma
            REAL(KIND(0d0)), INTENT(OUT) :: pi_inf
            
            ! Generic loop iterator
            INTEGER :: i
            
            
            ! Computing the density, the specific heat ratio function and the
            ! liquid stiffness function, respectively
            IF(adv_alphan) THEN
                
                rho = 0d0; gamma = 0d0; pi_inf = 0d0
                
                DO i = 1, num_fluids
                    rho    = rho    + q_vf(i)%sf(j,k,l)
                    gamma  = gamma  + q_vf(i+E_idx)%sf(j,k,l)*fluid_pp(i)%gamma
                    pi_inf = pi_inf + q_vf(i+E_idx)%sf(j,k,l)*fluid_pp(i)%pi_inf
                END DO
                
            ELSE
                
                rho    = q_vf(num_fluids)%sf(j,k,l)
                gamma  = fluid_pp(num_fluids)%gamma
                pi_inf = fluid_pp(num_fluids)%pi_inf
                
                DO i = 1, num_fluids-1
                    rho    = rho    + q_vf(i)%sf(j,k,l)
                    gamma  = gamma  + q_vf(i+E_idx)%sf(j,k,l)       &
                                    * ( fluid_pp(     i    )%gamma  &
                                      - fluid_pp(num_fluids)%gamma  )
                    pi_inf = pi_inf + q_vf(i+E_idx)%sf(j,k,l)       &
                                    * ( fluid_pp(     i    )%pi_inf &
                                      - fluid_pp(num_fluids)%pi_inf )
                END DO
                
            END IF
            
            
        END SUBROUTINE s_convert_species_to_mixture_variables ! ----------------
        
        
        
        
        
        SUBROUTINE s_initialize_variables_conversion_module() ! -------------------
        ! Description: Computation of parameters, allocation procedures, and/or
        !              any other tasks needed to properly setup the module
            
            
            ! Depending on the model selection for the equations of motion, the
            ! appropriate procedure for the conversion to the mixture variables
            ! is targeted by the procedure pointer
            
            IF (model_eqns == 1) THEN        ! Gamma/pi_inf model
                s_convert_to_mixture_variables => &
                            s_convert_mixture_to_mixture_variables
                
            ELSE IF (bubbles) THEN !SHB bubbles
                 s_convert_to_mixture_variables => &
                            s_convert_species_to_mixture_variables_bubbles
            ELSE
                ! Volume fraction model
                print*, 'convert species to mixture variables without bubbles'
                s_convert_to_mixture_variables => &
                            s_convert_species_to_mixture_variables
            END IF
            
            
        END SUBROUTINE s_initialize_variables_conversion_module ! -----------------
        
        
        
        
        
        SUBROUTINE s_convert_conservative_to_primitive_variables( q_cons_vf, &
                                                                  q_prim_vf  )
        ! Description: Converts the conservative variables to the primitive ones
            
            
            ! Conservative variables
            TYPE(scalar_field), &
            DIMENSION(sys_size), &
            INTENT(IN) :: q_cons_vf
            
            ! Primitive variables
            TYPE(scalar_field), &
            DIMENSION(sys_size), &
            INTENT(INOUT) :: q_prim_vf
            
            ! Density, specific heat ratio function, liquid stiffness function
            ! and dynamic pressure, as defined in the incompressible flow sense,
            ! respectively
            REAL(KIND(0d0)) :: rho
            REAL(KIND(0d0)) :: gamma
            REAL(KIND(0d0)) :: pi_inf
            REAL(KIND(0d0)) :: dyn_pres
            REAL(KIND(0d0)) :: nbub
            REAL(KIND(0d0)), dimension(nb) :: nRtmp
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l
            
            
            ! Converting the conservative variables to the primitive variables
            DO l = 0, p
                DO k = 0, n
                    DO j = 0, m
                        
                        ! Obtaining the density, specific heat ratio function
                        ! and the liquid stiffness function, respectively
                        if (model_eqns .ne. 4 ) then
                            CALL s_convert_to_mixture_variables( q_cons_vf,j,k,l, &
                                                             rho,gamma,pi_inf )
                        end if
                        
                        ! Transferring the continuity equation(s) variable(s)
                        DO i = 1, cont_idx%end
                            q_prim_vf(i)%sf(j,k,l) = q_cons_vf(i)%sf(j,k,l)
                        END DO
                        
                        ! Zeroing out the dynamic pressure since it is computed
                        ! iteratively by cycling through the momentum equations
                        dyn_pres = 0d0
 
                        ! Computing velocity and dynamic pressure from momenta
                        DO i = mom_idx%beg, mom_idx%end
                            if (model_eqns .ne. 4 ) then
                                q_prim_vf(i)%sf(j,k,l) = q_cons_vf(i)%sf(j,k,l)/rho
                                dyn_pres = dyn_pres + q_cons_vf(i)%sf(j,k,l) * &
                                                  q_prim_vf(i)%sf(j,k,l) / 2d0
                            else 
                                q_prim_vf(i)%sf(j,k,l) = q_cons_vf(i)%sf(j,k,l) &
                                                       / q_cons_vf(1)%sf(j,k,l)
                            end if
                        END DO

                        if ( (model_eqns .ne. 4) .and. (bubbles .neqv. .TRUE.) ) then
                            ! Computing the pressure from the energy
                            q_prim_vf(E_idx)%sf(j,k,l) = &
                                (q_cons_vf(E_idx)%sf(j,k,l)-dyn_pres-pi_inf)/gamma
                        else if ( (model_eqns .ne. 4) .and. bubbles) then
                            print*, 'getting model_eqns 2 with bubbles. cons to prim'
                            ! p = ( E/(1-alf) - 0.5 rho u u/(1-alf) - pi_inf_k )/gamma_k
                            q_prim_vf(E_idx)%sf(j,k,l) = &
                               ((q_cons_vf(E_idx)%sf(j,k,l)-dyn_pres)/(1.d0 - q_cons_vf(alf_idx)%sf(j,k,l)) &
                                                - pi_inf )/gamma                     
                        else
                            ! Tait EOS
                            ! p = (pl0 + pi_infty)(rho/(rho_l0(1-alf)))^gamma - pi_infty
                            q_prim_vf(E_idx)%sf(j,k,l) =                       & 
                                   (pref+fluid_pp(1)%pi_inf) *                  &
                                   (                                            & 
                                   q_prim_vf(1)%sf(j,k,l)/                     &
                                   (rhoref*(1-q_prim_vf(alf_idx)%sf(j,k,l)))   & 
                                   ) ** (1/fluid_pp(1)%gamma + 1) - fluid_pp(1)%pi_inf
                        end if

                        ! Set partial pressures to mixture pressure
                        IF(model_eqns == 3) THEN
                            DO i = internalEnergies_idx%beg, internalEnergies_idx%end
                                q_prim_vf(i)%sf(j,k,l) = q_prim_vf(E_idx)%sf(j,k,l)
                            END DO
                        ENDIF

                        ! Transfer the advection equation(s) variable(s)
                        DO i = adv_idx%beg, adv_idx%end
                            q_prim_vf(i)%sf(j,k,l) = q_cons_vf(i)%sf(j,k,l)
                        END DO

                        ! \phi = (n\phi)/n  (n = nbub)
                        if (bubbles) then
                            ! compute n from conserved variables
                            ! n = sqrt( 4pi/(3 alpha) * \bar{nR}**3 )
                            do i = 1,nb
                                nRtmp(i) = q_cons_vf(bub_idx%rs(i))%sf(j,k,l)
                            end do
                            call s_comp_n_from_cons( q_cons_vf(alf_idx)%sf(j,k,l), nRtmp, nbub)
                            do i = bub_idx%beg, sys_size
                                q_prim_vf(i)%sf(j,k,l) = q_cons_vf(i)%sf(j,k,l)/nbub
                            end do
                        end if
                    END DO
                END DO
            END DO
            
            
        END SUBROUTINE s_convert_conservative_to_primitive_variables ! ---------
        
        ! Used when initializing patch 
        SUBROUTINE s_convert_primitive_to_conservative_variables( q_prim_vf, &
                                                                  q_cons_vf  )
        ! Description: Converts the primitive variables to the conservative ones
            
            ! Primitive variables
            TYPE(scalar_field), &
            DIMENSION(sys_size), &
            INTENT(IN) :: q_prim_vf
            
            ! Conservative variables
            TYPE(scalar_field), &
            DIMENSION(sys_size), &
            INTENT(INOUT) :: q_cons_vf
            
            ! Density, specific heat ratio function, liquid stiffness function
            ! and dynamic pressure, as defined in the incompressible flow sense,
            ! respectively
            REAL(KIND(0d0)) :: rho
            REAL(KIND(0d0)) :: gamma
            REAL(KIND(0d0)) :: pi_inf
            REAL(KIND(0d0)) :: dyn_pres
            REAL(KIND(0d0)) :: nbub
            REAL(KIND(0d0)), dimension(nb) :: Rtmp

            ! Generic loop iterators
            INTEGER :: i,j,k,l
            
            ! Converting the primitive variables to the conservative variables
            DO l = 0, p
                DO k = 0, n
                    DO j = 0, m
                        
                        ! Obtaining the density, specific heat ratio function
                        ! and the liquid stiffness function, respectively
                        CALL s_convert_to_mixture_variables( q_prim_vf,j,k,l, &
                                                             rho,gamma,pi_inf )
                        
                        ! Transferring the continuity equation(s) variable(s)
                        DO i = 1, cont_idx%end
                            q_cons_vf(i)%sf(j,k,l) = q_prim_vf(i)%sf(j,k,l)
                        END DO
                        
                        ! Zeroing out the dynamic pressure since it is computed
                        ! iteratively by cycling through the velocity equations
                        dyn_pres = 0d0
                        
                        ! Computing momenta and dynamic pressure from velocity
                        DO i = mom_idx%beg, mom_idx%end
                            q_cons_vf(i)%sf(j,k,l) = rho*q_prim_vf(i)%sf(j,k,l)
                            dyn_pres = dyn_pres + q_cons_vf(i)%sf(j,k,l) * &
                                                  q_prim_vf(i)%sf(j,k,l) / 2d0
                        END DO
                        
                        ! Computing the energy from the pressure
                        if ( (model_eqns .ne. 4) .and. (bubbles .neqv. .TRUE.) ) then
                            ! E = Gamma*P + \rho u u /2 + \pi_inf
                            q_cons_vf(E_idx)%sf(j,k,l) = &
                                gamma*q_prim_vf(E_idx)%sf(j,k,l)+dyn_pres+pi_inf
                        else if ( (model_eqns .ne. 4) .and. (bubbles) ) then
                            ! \tilde{E} = dyn_pres + (1-\alf)(\Gamma p_l + \Pi_inf)
                            q_cons_vf(E_idx)%sf(j,k,l) = dyn_pres +         &
                                (1.d0 - q_prim_vf(alf_idx)%sf(j,k,l)) *     &    
                                (gamma*q_prim_vf(E_idx)%sf(j,k,l) + pi_inf)
                        else
                            !Tait EOS, no conserved energy variable
                            q_cons_vf(E_idx)%sf(j,k,l) = 0.
                        end if
                        !print*, 'energy = ', j,q_cons_vf(E_idx)%sf(j,k,l)

                        ! Computing the internal energies from the pressure and continuities
                        IF(model_eqns == 3) THEN
                            DO i = internalEnergies_idx%beg, internalEnergies_idx%end
                                q_cons_vf(i)%sf(j,k,l) = q_cons_vf(i-adv_idx%end)%sf(j,k,l) * & 
                                    fluid_pp(i-adv_idx%end)%gamma*q_prim_vf(E_idx)%sf(j,k,l)+fluid_pp(i-adv_idx%end)%pi_inf
                            END DO
                        ENDIF

                        ! Transferring the advection equation(s) variable(s)
                        DO i = adv_idx%beg, adv_idx%end
                            q_cons_vf(i)%sf(j,k,l) = q_prim_vf(i)%sf(j,k,l)
                        END DO

                        ! n\phi = n*\phi  (n = nbub)
                        if (bubbles) then
                            ! n = 3 alpha / (4 pi \bar{R^3})
                            do i = 1,nb
                                Rtmp(i) = q_prim_vf(bub_idx%rs(i))%sf(j,k,l)
                            end do
                            call s_comp_n_from_prim( q_prim_vf(alf_idx)%sf(j,k,l), Rtmp, nbub)

                            do i = bub_idx%beg, sys_size
                                q_cons_vf(i)%sf(j,k,l) = q_prim_vf(i)%sf(j,k,l)*nbub
                            end do
                        end if
                        
                    END DO
                END DO
            END DO
            
        END SUBROUTINE s_convert_primitive_to_conservative_variables ! ---------
        
        
        SUBROUTINE s_finalize_variables_conversion_module() ! ----------------
        ! Description: Deallocation procedures for the module
            
            
            ! Nullifying the procedure pointer to the subroutine transfering/
            ! computing the mixture/species variables to the mixture variables
            s_convert_to_mixture_variables => NULL()
            
            
        END SUBROUTINE s_finalize_variables_conversion_module ! --------------
        
        
END MODULE m_variables_conversion