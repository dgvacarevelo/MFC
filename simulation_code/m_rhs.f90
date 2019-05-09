! MFC v3.0 - Simulation Code: m_rhs.f90
! Description: The module contains the subroutines used to calculate the right-
!              hand-side (RHS) in the quasi-conservative, shock- and interface-
!              capturing finite-volume framework for the multicomponent Navier-
!              Stokes equations supplemented by appropriate advection equations
!              used to capture the material interfaces. The system of equations
!              is closed by the stiffened gas equation of state, as well as any
!              required mixture relationships. Capillarity effects are included
!              and are modeled by the means of a volume force acting across the
!              diffuse material interface region. The implementation details of
!              surface tension may be found in Perigaud and Saurel (2005). Note
!              that both viscous and surface tension effects are only available
!              in the volume fraction model.
! Author: Vedran Coralic
! Date: 07/12/12


MODULE m_rhs
    
    
    ! Dependencies =============================================================
    USE m_derived_types        ! Definitions of the derived types
    
    USE m_global_parameters    ! Definitions of the global parameters
    
    USE m_mpi_proxy            ! Message passing interface (MPI) module proxy
    
    USE m_variables_conversion ! State variables type conversion procedures
    
    USE m_weno                 ! Weighted and essentially non-oscillatory (WENO)
                               ! schemes for spatial reconstruction of variables
    
    USE m_riemann_solvers      ! Exact and approximate Riemann problem solvers
    
    USE m_cbc                  ! Characteristic boundary conditions (CBC)

    use m_bubbles
    ! ==========================================================================
    
    
    IMPLICIT NONE
    
    PRIVATE; PUBLIC :: s_initialize_rhs_module, &
                       s_compute_rhs          , &
                       s_pressure_relaxation_procedure, &
                       s_populate_variables_buffers, &
                       s_finalize_rhs_module, &
                       s_get_crv, &
                       s_get_viscous
    
    
    ! This variable contains the WENO-reconstructed values of the cell-average
    ! conservative variables, which are located in q_cons_vf, at cell-interior
    ! Gaussian quadrature points (QP).
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: q_cons_qp
    
    ! The primitive variables at cell-interior Gaussian quadrature points. These
    ! are calculated from the conservative variables and gradient magnitude (GM)
    ! of the volume fractions, q_cons_qp and gm_alpha_qp, respectively.
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: q_prim_qp
    
    ! The left (L) and the right (R) WENO-reconstructed cell-boundary values,
    ! including cell-boundary Gaussian quadrature points, of the cell-average
    ! conservative variables. The latter are stored in the variable q_cons_qp
    ! (NDQP - normal direction quadrature points).
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: qL_cons_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: qR_cons_ndqp
    
    ! The left and right WENO-reconstructed cell-boundary values, that include
    ! cell-boundary Gaussian quadrature points, of the cell-averaged primitive
    ! variables. The latter are stored in the variable q_prim_qp.
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: qL_prim_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: qR_prim_ndqp
    
    ! The first-order spatial derivatives of the primitive variables at cell-
    ! interior Guassian quadrature points. These are WENO-reconstructed from
    ! their respective cell-average values, obtained through the application
    ! of the divergence theorem on the integral-average cell-boundary values
    ! of the primitive variables, located in qK_prim_ndqp, where K = L or R.
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: dq_prim_dx_qp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: dq_prim_dy_qp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: dq_prim_dz_qp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: gm_vel_qp
    
    ! The left and right WENO-reconstructed cell-boundary values of the cell-
    ! average first-order spatial derivatives of the primitive variables. The
    ! cell-average of the first-order spatial derivatives may be found in the
    ! variables dq_prim_ds_qp, where s = x, y or z.
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: dqL_prim_dx_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: dqL_prim_dy_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: dqL_prim_dz_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: dqR_prim_dx_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: dqR_prim_dy_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: dqR_prim_dz_ndqp
    
    ! The gradient magnitude of the volume fractions at cell-interior Gaussian
    ! quadrature points. gm_alpha_qp is calculated from individual first-order
    ! spatial derivatives located in dq_prim_ds_qp.
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: gm_alpha_qp
    
    ! The left and right WENO-reconstructed cell-boundary values of the cell-
    ! average gradient magnitude of volume fractions, located in gm_alpha_qp.
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: gm_alphaL_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: gm_alphaR_ndqp
    
    ! The left and right cell-boundary values of the vector components of the
    ! unit normal (UN) of the volume fractions. These are directly calculated
    ! from the first-order spatial derivatives, stored in dqK_prim_ds_ndqp.
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: un_alphaL_x_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: un_alphaL_y_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: un_alphaL_z_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: un_alphaR_x_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: un_alphaR_y_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: un_alphaR_z_ndqp
    
    ! The cell-average curvature of volume fractions. kappa_vf is calculated by
    ! the divergence theorem using the integral-average cell-boundary values of
    ! of the components of the unit normals, which are located in the variables
    ! un_alphaK_s_ndqp.
    TYPE(scalar_field), ALLOCATABLE, DIMENSION(:) :: kappa_vf
    
    ! The left and right WENO-reconstructed cell-boundary values of the cell-
    ! average curvature of the volume fractions. The cell-average quantities
    ! are located in the variable kappa_vf.
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: kappaL_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: kappaR_ndqp
    
    ! The cell-boundary values of the fluxes (src - source, gsrc - geometrical
    ! source). These are computed by applying the chosen Riemann problem solver
    ! on the left and right cell-boundary values of the primitive variables, 
    ! qK_prim_ndqp, the first-order spatial derivatives, dqK_prim_ds_ndqp, as
    ! well as the curvature of volume fractions, kappaK_ndqp.
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: flux_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: flux_src_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: flux_gsrc_ndqp

    ! Additional vector and scalar fields needed in alternate computation
    ! of the cell-average curvature of volume fractions.
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: dgm_alpha_dx_qp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: dgm_alpha_dy_qp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: dgm_alpha_dz_qp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: un_alpha_x_qp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: un_alpha_y_qp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: un_alpha_z_qp

    TYPE(scalar_field), ALLOCATABLE, DIMENSION(:) :: laplacian_vf
    TYPE(scalar_field), ALLOCATABLE, DIMENSION(:) :: alt_kappa_vf

    ! Additional field for regularization terms
    TYPE(scalar_field), ALLOCATABLE, DIMENSION(:) :: reg_src_vf

    ! Additional field for capillary source terms
    REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:,:,:) :: We_mtm_src
    REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:,:,:) :: We_nrg_src
    TYPE(scalar_field), ALLOCATABLE, DIMENSION(:) :: tau_We_vf
    TYPE(scalar_field), ALLOCATABLE, DIMENSION(:) :: tau_Re_vf

    ! Additional variables for applying a flux limiter to the advection
    ! equation
    REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:,:,:) :: flux_lim_func
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: lo_flux_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: lo_flux_src_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: lo_flux_gsrc_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: hi_flux_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: hi_flux_src_ndqp
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: hi_flux_gsrc_ndqp
    
    ! The indical bounds in the coordinate directions of the Gaussian numerical
    ! integration that identify the locations of the quadrature points that are
    ! necessary to evaluate cell-interior integral-averages.
    TYPE(bounds_info) :: ieta,iksi,itau
    
    ! The indical bounds in the coordinate directions of the Gaussian numerical
    ! integration that identify the locations of the quadrature points that are
    ! necessary to evaluate cell-boundary integral-averages.
    TYPE(bounds_info) :: ichi,ipsi
    
    ! Vector field indical bounds
    TYPE(bounds_info) :: iv
    
    ! Indical bounds in the x-, y- and z-directions
    TYPE(bounds_info) :: ix,iy,iz
   
    ! source terms for bubbles
    REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:,:,:) :: bub_adv_src
    REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:,:,:,:) :: bub_r_src, bub_v_src, bub_p_src, bub_m_src
    ! matrix for \div(u)
    TYPE(scalar_field) :: divu
    !REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:,:,:) :: divu
 
    ! source terms for monopole
    REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:,:,:) :: mono_mass_src, mono_e_src
    REAL(KIND(0d0)), ALLOCATABLE, DIMENSION(:,:,:,:) :: mono_mom_src
    
    TYPE(vector_field), ALLOCATABLE, DIMENSION(:,:,:) :: myflux_vf, myflux_src_vf
    TYPE(scalar_field) :: alf_sum


    character(50) :: file_path
    
    CONTAINS
        
        
        SUBROUTINE s_initialize_rhs_module() ! ---------------------------------
        ! Description: The computation of parameters, the allocation of memory,
        !              the association of pointers and/or the execution of any
        !              other procedures that are necessary to setup the module.
            
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l
            
            
            ! Configuring Cell-Interior Quadrature Points ======================
            ieta%beg = 0; iksi%beg = 0; itau%beg = 0
            
            IF(commute_err) THEN
                ieta%beg = -1; IF(n > 0) iksi%beg = -1; IF(p > 0) itau%beg = -1
            END IF
            
            ieta%end = -ieta%beg; iksi%end = -iksi%beg; itau%end = -itau%beg
            ! ==================================================================
            
            
            ! Configuring Cell-Boundary Quadrature Points ======================
            ichi%beg = 0; ipsi%beg = 0
            
            IF(split_err) THEN
                ichi%beg = -1; IF(p > 0) ipsi%beg = -1
            END IF
            
            ichi%end = -ichi%beg; ipsi%end = -ipsi%beg
            ! ==================================================================
            
            
            ! Configuring Coordinate Direction Indexes =========================
            ix%beg = -buff_size; iy%beg = 0; iz%beg = 0
            
            IF(n > 0) iy%beg = -buff_size; IF(p > 0) iz%beg = -buff_size
            
            ix%end = m - ix%beg; iy%end = n - iy%beg; iz%end = p - iz%beg
            ! ==================================================================
            
            IF (We_size > 0 .AND. We_rhs_flux) THEN
                ALLOCATE(tau_We_vf(1:sys_size))
                DO i = 1, num_dims
                    ALLOCATE(tau_We_vf(cont_idx%end+i)%sf(ix%beg:ix%end, &
                                                          iy%beg:iy%end, &
                                                          iz%beg:iz%end ))
                END DO
                ALLOCATE(tau_We_vf(E_idx)%sf(ix%beg:ix%end, &
                                             iy%beg:iy%end, &
                                             iz%beg:iz%end ))
            END IF

            IF (ANY(Re_size > 0) .AND. cyl_coord) THEN
                ALLOCATE(tau_Re_vf(1:sys_size))
                DO i = 1, num_dims
                    ALLOCATE(tau_Re_vf(cont_idx%end+i)%sf(ix%beg:ix%end, &
                                                          iy%beg:iy%end, &
                                                          iz%beg:iz%end ))
                END DO
                ALLOCATE(tau_Re_vf(E_idx)%sf(ix%beg:ix%end, &
                                             iy%beg:iy%end, &
                                             iz%beg:iz%end ))
            END IF
            
            ! Allocation/Association of q_cons_qp and q_prim_qp ================
            ALLOCATE(q_cons_qp( ieta%beg:ieta%end, &
                                iksi%beg:iksi%end, &
                                itau%beg:itau%end ))
            ALLOCATE(q_prim_qp( ieta%beg:ieta%end, &
                                iksi%beg:iksi%end, &
                                itau%beg:itau%end ))
            
            DO k = itau%beg, itau%end
                DO j = iksi%beg, iksi%end
                    DO i = ieta%beg, ieta%end
                     
                        ALLOCATE(q_cons_qp(i,j,k)%vf(1:sys_size))
                        ALLOCATE(q_prim_qp(i,j,k)%vf(1:sys_size))
                        
                        IF(i /= 0 .AND. ABS(j) >= ABS(k)) THEN
                           
                            DO l = 1, sys_size
                                ALLOCATE(q_cons_qp(i,j,k)%vf(l)%sf( ix%beg:ix%end, &
                                                                    iy%beg:iy%end, &
                                                                    iz%beg:iz%end ))
                                q_prim_qp(i,j,k)%vf(l)%sf => &
                                q_cons_qp(i,j,k)%vf(l)%sf
                            END DO
                            
                            DO l = mom_idx%beg, E_idx
                                ALLOCATE(q_prim_qp(i,j,k)%vf(l)%sf( ix%beg:ix%end, &
                                                                    iy%beg:iy%end, &
                                                                    iz%beg:iz%end ))
                            END DO
                            IF (model_eqns == 3) THEN
                                DO l = internalEnergies_idx%beg, internalEnergies_idx%end
                                    ALLOCATE(q_prim_qp(i,j,k)%vf(l)%sf( ix%beg:ix%end, &
                                                                        iy%beg:iy%end, &
                                                                        iz%beg:iz%end ))
                                END DO
                            END IF
       
                            if (bubbles) then
                                DO l = bub_idx%beg,sys_size
                                    ALLOCATE(q_prim_qp(i,j,k)%vf(l)%sf( ix%beg:ix%end, &
                                                                        iy%beg:iy%end, &
                                                                        iz%beg:iz%end ))
                                END DO
                            end if
                        END IF
                     
                    END DO
                END DO
            END DO
            ! ==================================================================
            
            
            ! Allocation/Association of qK_cons_ndqp and qK_prim_ndqp ==========
            ALLOCATE(qL_cons_ndqp(     1   :num_dims, &
                                   ichi%beg:ichi%end, &
                                   ipsi%beg:ipsi%end ))
            ALLOCATE(qR_cons_ndqp(     1   :num_dims, &
                                   ichi%beg:ichi%end, &
                                   ipsi%beg:ipsi%end ))
            ALLOCATE(qL_prim_ndqp(     1   :num_dims, &
                                   ichi%beg:ichi%end, &
                                   ipsi%beg:ipsi%end ))
            ALLOCATE(qR_prim_ndqp(     1   :num_dims, &
                                   ichi%beg:ichi%end, &
                                   ipsi%beg:ipsi%end ))
 
            ALLOCATE(myflux_vf(     1   :num_dims, &
                                   ichi%beg:ichi%end, &
                                   ipsi%beg:ipsi%end ))
            ALLOCATE(myflux_src_vf(     1   :num_dims, &
                                   ichi%beg:ichi%end, &
                                   ipsi%beg:ipsi%end ))

            allocate( alf_sum%sf( &
                                    ix%beg:ix%end, &
                                    iy%beg:iy%end, &
                                    iz%beg:iz%end ))

            DO k = ipsi%beg, ipsi%end
                DO j = ichi%beg, ichi%end
                    DO i = 1, num_dims
                     
                        ALLOCATE(qL_cons_ndqp(i,j,k)%vf(1:sys_size))
                        ALLOCATE(qR_cons_ndqp(i,j,k)%vf(1:sys_size))
                        ALLOCATE(qL_prim_ndqp(i,j,k)%vf(1:sys_size))
                        ALLOCATE(qR_prim_ndqp(i,j,k)%vf(1:sys_size))

                        ALLOCATE(myflux_vf(i,j,k)%vf(1:sys_size))
                        ALLOCATE(myflux_src_vf(i,j,k)%vf(1:sys_size))


                        IF(ABS(j) >= ABS(k)) THEN
                            do l = 1,sys_size
                                ALLOCATE(myflux_vf(i,j,k)%vf(l)%sf( &
                                                            ix%beg:ix%end, &
                                                            iy%beg:iy%end, &
                                                            iz%beg:iz%end ))
                                ALLOCATE(myflux_src_vf(i,j,k)%vf(l)%sf( &
                                                            ix%beg:ix%end, &
                                                            iy%beg:iy%end, &
                                                            iz%beg:iz%end ))
                            end do
 
                            IF(i == 1) THEN
                              
                                DO l = 1, cont_idx%end
                                    ALLOCATE(qL_cons_ndqp(i,j,k)%vf(l)%sf( &
                                                            ix%beg:ix%end, &
                                                            iy%beg:iy%end, &
                                                            iz%beg:iz%end ))
                                    ALLOCATE(qR_cons_ndqp(i,j,k)%vf(l)%sf( &
                                                            ix%beg:ix%end, &
                                                            iy%beg:iy%end, &
                                                            iz%beg:iz%end ))
                                END DO
                              
                                IF(weno_vars == 1) THEN
                                    DO l = mom_idx%beg, E_idx
                                        ALLOCATE(qL_cons_ndqp(i,j,k)%vf(l)%sf( &
                                                                ix%beg:ix%end, &
                                                                iy%beg:iy%end, &
                                                                iz%beg:iz%end ))
                                        ALLOCATE(qR_cons_ndqp(i,j,k)%vf(l)%sf( &
                                                                ix%beg:ix%end, &
                                                                iy%beg:iy%end, &
                                                                iz%beg:iz%end ))
                                    END DO
                                END IF
                              
                                DO l = mom_idx%beg, E_idx
                                    ALLOCATE(qL_prim_ndqp(i,j,k)%vf(l)%sf( &
                                                            ix%beg:ix%end, &
                                                            iy%beg:iy%end, &
                                                            iz%beg:iz%end ))
                                    ALLOCATE(qR_prim_ndqp(i,j,k)%vf(l)%sf( &
                                                            ix%beg:ix%end, &
                                                            iy%beg:iy%end, &
                                                            iz%beg:iz%end ))
                                END DO
                                IF (model_eqns == 3) THEN
                                    DO l = internalEnergies_idx%beg, internalEnergies_idx%end
                                        ALLOCATE(qL_prim_ndqp(i,j,k)%vf(l)%sf( &
                                                                ix%beg:ix%end, &
                                                                iy%beg:iy%end, &
                                                                iz%beg:iz%end ))
                                        ALLOCATE(qR_prim_ndqp(i,j,k)%vf(l)%sf( &
                                                                ix%beg:ix%end, &
                                                                iy%beg:iy%end, &
                                                                iz%beg:iz%end ))
                                    END DO
                                END IF
                             
                                DO l = adv_idx%beg, sys_size
                                        ALLOCATE(qL_cons_ndqp(i,j,k)%vf(l)%sf( &
                                                        ix%beg:ix%end, &
                                                        iy%beg:iy%end, &
                                                        iz%beg:iz%end ))
                                        ALLOCATE(qR_cons_ndqp(i,j,k)%vf(l)%sf( &
                                                        ix%beg:ix%end, &
                                                        iy%beg:iy%end, &
                                                        iz%beg:iz%end ))
                                END DO

                                if (bubbles) then
                                    DO l = bub_idx%beg, sys_size
                                        ALLOCATE(qL_prim_ndqp(i,j,k)%vf(l)%sf( &
                                                            ix%beg:ix%end, &
                                                            iy%beg:iy%end, &
                                                            iz%beg:iz%end ))
                                        ALLOCATE(qR_prim_ndqp(i,j,k)%vf(l)%sf( &
                                                            ix%beg:ix%end, &
                                                            iy%beg:iy%end, &
                                                            iz%beg:iz%end ))
                                    END DO
                                end if
                              
                            ELSE
                              
                                DO l = 1, sys_size
                                    qL_cons_ndqp(i,j,k)%vf(l)%sf => &
                                    qL_cons_ndqp(1,j,k)%vf(l)%sf
                                    qR_cons_ndqp(i,j,k)%vf(l)%sf => &
                                    qR_cons_ndqp(1,j,k)%vf(l)%sf
                                    qL_prim_ndqp(i,j,k)%vf(l)%sf => &
                                    qL_prim_ndqp(1,j,k)%vf(l)%sf
                                    qR_prim_ndqp(i,j,k)%vf(l)%sf => &
                                    qR_prim_ndqp(1,j,k)%vf(l)%sf
                                END DO
                              
                                IF( (char_decomp .NEQV. .TRUE.) &
                                               .AND.            &
                                       ANY(Re_size > 0)       ) THEN
                                    IF(weno_vars == 1) THEN
                                        DO l = 1, mom_idx%end
                                            ALLOCATE(qL_cons_ndqp(i,j,k)%vf(l)%sf( &
                                                                    ix%beg:ix%end, &
                                                                    iy%beg:iy%end, &
                                                                    iz%beg:iz%end ))
                                            ALLOCATE(qR_cons_ndqp(i,j,k)%vf(l)%sf( &
                                                                    ix%beg:ix%end, &
                                                                    iy%beg:iy%end, &
                                                                    iz%beg:iz%end ))
                                        END DO
                                    ELSE
                                        DO l = mom_idx%beg, mom_idx%end
                                            ALLOCATE(qL_prim_ndqp(i,j,k)%vf(l)%sf( &
                                                                    ix%beg:ix%end, &
                                                                    iy%beg:iy%end, &
                                                                    iz%beg:iz%end ))
                                            ALLOCATE(qR_prim_ndqp(i,j,k)%vf(l)%sf( &
                                                                    ix%beg:ix%end, &
                                                                    iy%beg:iy%end, &
                                                                    iz%beg:iz%end ))
                                        END DO
                                        IF (model_eqns == 3) THEN
                                            DO l = internalEnergies_idx%beg, internalEnergies_idx%end
                                                ALLOCATE(qL_prim_ndqp(i,j,k)%vf(l)%sf( &
                                                                        ix%beg:ix%end, &
                                                                        iy%beg:iy%end, &
                                                                        iz%beg:iz%end ))
                                                ALLOCATE(qR_prim_ndqp(i,j,k)%vf(l)%sf( &
                                                                        ix%beg:ix%end, &
                                                                        iy%beg:iy%end, &
                                                                        iz%beg:iz%end ))
                                            END DO
                                        END IF
                                    END IF
                                END IF
                              
                                DO l = 1, crv_size
                                    ALLOCATE(qL_cons_ndqp(i,j,k)%vf( &
                                               E_idx+crv_idx(l))%sf( &
                                                      ix%beg:ix%end, &
                                                      iy%beg:iy%end, &
                                                      iz%beg:iz%end ))
                                    ALLOCATE(qR_cons_ndqp(i,j,k)%vf( &
                                               E_idx+crv_idx(l))%sf( &
                                                      ix%beg:ix%end, &
                                                      iy%beg:iy%end, &
                                                      iz%beg:iz%end ))
                                END DO
                              
                            END IF
                           
                            DO l = 1, cont_idx%end
                                qL_prim_ndqp(i,j,k)%vf(l)%sf => &
                                qL_cons_ndqp(i,j,k)%vf(l)%sf
                                qR_prim_ndqp(i,j,k)%vf(l)%sf => &
                                qR_cons_ndqp(i,j,k)%vf(l)%sf
                            END DO
                           
                            !DO l = adv_idx%beg, adv_idx%end
                            !    qL_prim_ndqp(i,j,k)%vf(l)%sf => &
                            !    qL_cons_ndqp(i,j,k)%vf(l)%sf
                            !    qR_prim_ndqp(i,j,k)%vf(l)%sf => &
                            !    qR_cons_ndqp(i,j,k)%vf(l)%sf
                            !END DO


                            !SHB changes
                            if (adv_alphan ) then
                                DO l = adv_idx%beg, adv_idx%end
                                    qL_prim_ndqp(i,j,k)%vf(l)%sf => &
                                    qL_cons_ndqp(i,j,k)%vf(l)%sf
                                    qR_prim_ndqp(i,j,k)%vf(l)%sf => &
                                    qR_cons_ndqp(i,j,k)%vf(l)%sf
                                END DO
                            else
                                DO l = adv_idx%beg, adv_idx%end+1    
                                    qL_prim_ndqp(i,j,k)%vf(l)%sf => &
                                    qL_cons_ndqp(i,j,k)%vf(l)%sf
                                    qR_prim_ndqp(i,j,k)%vf(l)%sf => &
                                    qR_cons_ndqp(i,j,k)%vf(l)%sf
                                END DO
                            end if
                           
                        END IF
                     
                    END DO
                END DO
            END DO
            ! END: Allocation/Association of qK_cons_ndqp and qK_prim_ndqp =====

    
            ! Allocation of dq_prim_ds_qp ======================================
            IF(ANY(Re_size > 0) .OR. We_size > 0 .OR. tvd_rhs_flux) THEN
               
                ALLOCATE(dq_prim_dx_qp( ieta%beg:ieta%end, &
                                        iksi%beg:iksi%end, &
                                        itau%beg:itau%end ))
                ALLOCATE(dq_prim_dy_qp( ieta%beg:ieta%end, &
                                        iksi%beg:iksi%end, &
                                        itau%beg:itau%end ))
                ALLOCATE(dq_prim_dz_qp( ieta%beg:ieta%end, &
                                        iksi%beg:iksi%end, &
                                        itau%beg:itau%end ))
                ALLOCATE(    gm_vel_qp( ieta%beg:ieta%end, &
                                        iksi%beg:iksi%end, &
                                        itau%beg:itau%end ))
                
                DO k = itau%beg, itau%end
                    DO j = iksi%beg, iksi%end
                        DO i = ieta%beg, ieta%end
                         
                            ALLOCATE(dq_prim_dx_qp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(dq_prim_dy_qp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(dq_prim_dz_qp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(    gm_vel_qp(i,j,k)%vf(1:sys_size))
                         
                            IF(ALL((/i,j,k/) == 0) .AND. ANY(Re_size > 0)) THEN
                            
                                DO l = mom_idx%beg, mom_idx%end
                                    ALLOCATE(dq_prim_dx_qp(i,j,k)%vf(l)%sf( &
                                                             ix%beg:ix%end, &
                                                             iy%beg:iy%end, &
                                                             iz%beg:iz%end ))
                                    ALLOCATE(    gm_vel_qp(i,j,k)%vf(l)%sf( &
                                                             ix%beg:ix%end, &
                                                             iy%beg:iy%end, &
                                                             iz%beg:iz%end ))
                                END DO
                            
                                IF(n > 0) THEN
                               
                                    DO l = mom_idx%beg, mom_idx%end
                                        ALLOCATE(dq_prim_dy_qp(i,j,k)%vf(l)%sf( &
                                                                 ix%beg:ix%end, &
                                                                 iy%beg:iy%end, &
                                                                 iz%beg:iz%end ))
                                    END DO
                               
                                    IF(p > 0) THEN
                                        DO l = mom_idx%beg, mom_idx%end
                                            ALLOCATE(dq_prim_dz_qp(i,j,k)%vf(l)%sf( &
                                                                     ix%beg:ix%end, &
                                                                     iy%beg:iy%end, &
                                                                     iz%beg:iz%end ))
                                        END DO
                                    END IF
                               
                                END IF
                            
                            END IF
                         
                            IF( (i /= 0 .AND. ABS(j) >= ABS(k)) &
                                              .OR.              &
                                   ALL((/i,j,k/) == 0)       ) THEN
                            
                                DO l = 1, crv_size
                                    ALLOCATE(dq_prim_dx_qp(i,j,k)%vf( &
                                                E_idx+crv_idx(l))%sf( &
                                                       ix%beg:ix%end, &
                                                       iy%beg:iy%end, &
                                                       iz%beg:iz%end ))
                                    IF (n > 0) THEN
                                        ALLOCATE(dq_prim_dy_qp(i,j,k)%vf( &
                                                    E_idx+crv_idx(l))%sf( &
                                                           ix%beg:ix%end, &
                                                           iy%beg:iy%end, &
                                                           iz%beg:iz%end ))
                                    END IF
                                END DO
                            
                                IF(p > 0) THEN
                                    DO l = 1, crv_size
                                        ALLOCATE(dq_prim_dz_qp(i,j,k)%vf( &
                                                    E_idx+crv_idx(l))%sf( &
                                                           ix%beg:ix%end, &
                                                           iy%beg:iy%end, &
                                                           iz%beg:iz%end ))
                                    END DO
                                END IF
                            
                            END IF
                         
                        END DO
                    END DO
                END DO
               
            END IF
            ! END: Allocation of dq_prim_ds_qp =================================
            
            
            ! Allocation/Association of dqK_prim_ds_ndqp =======================
            ALLOCATE(dqL_prim_dx_ndqp(     1   :num_dims, &
                                       ichi%beg:ichi%end, &
                                       ipsi%beg:ipsi%end ))
            ALLOCATE(dqL_prim_dy_ndqp(     1   :num_dims, &
                                       ichi%beg:ichi%end, &
                                       ipsi%beg:ipsi%end ))
            ALLOCATE(dqL_prim_dz_ndqp(     1   :num_dims, &
                                       ichi%beg:ichi%end, &
                                       ipsi%beg:ipsi%end ))
            ALLOCATE(dqR_prim_dx_ndqp(     1   :num_dims, &
                                       ichi%beg:ichi%end, &
                                       ipsi%beg:ipsi%end ))
            ALLOCATE(dqR_prim_dy_ndqp(     1   :num_dims, &
                                       ichi%beg:ichi%end, &
                                       ipsi%beg:ipsi%end ))
            ALLOCATE(dqR_prim_dz_ndqp(     1   :num_dims, &
                                       ichi%beg:ichi%end, &
                                       ipsi%beg:ipsi%end ))
            
            IF(ANY(Re_size > 0) .OR. We_size > 0) THEN
                DO k = ipsi%beg, ipsi%end
                    DO j = ichi%beg, ichi%end
                        DO i = 1, num_dims
                        
                            ALLOCATE(dqL_prim_dx_ndqp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(dqL_prim_dy_ndqp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(dqL_prim_dz_ndqp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(dqR_prim_dx_ndqp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(dqR_prim_dy_ndqp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(dqR_prim_dz_ndqp(i,j,k)%vf(1:sys_size))
                            
                            IF(ABS(j) >= ABS(k)) THEN
                               
                                IF(ANY(Re_size > 0)) THEN

                                    DO l = mom_idx%beg, mom_idx%end
                                        ALLOCATE(dqL_prim_dx_ndqp(i,j,k)%vf(l)%sf( &
                                                                    ix%beg:ix%end, &
                                                                    iy%beg:iy%end, &
                                                                    iz%beg:iz%end ))
                                        ALLOCATE(dqR_prim_dx_ndqp(i,j,k)%vf(l)%sf( &
                                                                    ix%beg:ix%end, &
                                                                    iy%beg:iy%end, &
                                                                    iz%beg:iz%end ))
                                    END DO
                                  
                                    IF(n > 0) THEN
                                        DO l = mom_idx%beg, mom_idx%end
                                            ALLOCATE(dqL_prim_dy_ndqp(i,j,k)%vf(l)%sf( &
                                                                        ix%beg:ix%end, &
                                                                        iy%beg:iy%end, &
                                                                        iz%beg:iz%end ))
                                            ALLOCATE(dqR_prim_dy_ndqp(i,j,k)%vf(l)%sf( &
                                                                        ix%beg:ix%end, &
                                                                        iy%beg:iy%end, &
                                                                        iz%beg:iz%end ))
                                        END DO
                                    END IF
                                  
                                    IF(p > 0) THEN
                                        DO l = mom_idx%beg, mom_idx%end
                                            ALLOCATE(dqL_prim_dz_ndqp(i,j,k)%vf(l)%sf( &
                                                                        ix%beg:ix%end, &
                                                                        iy%beg:iy%end, &
                                                                        iz%beg:iz%end ))
                                            ALLOCATE(dqR_prim_dz_ndqp(i,j,k)%vf(l)%sf( &
                                                                        ix%beg:ix%end, &
                                                                        iy%beg:iy%end, &
                                                                        iz%beg:iz%end ))
                                        END DO
                                    END IF

                                END IF
                                  
                                DO l = 1, crv_size
                                    ALLOCATE(dqL_prim_dx_ndqp(i,j,k)%vf( &
                                                   E_idx+crv_idx(l))%sf( &
                                                          ix%beg:ix%end, &
                                                          iy%beg:iy%end, &
                                                          iz%beg:iz%end ))
                                    ALLOCATE(dqR_prim_dx_ndqp(i,j,k)%vf( &
                                                   E_idx+crv_idx(l))%sf( &
                                                          ix%beg:ix%end, &
                                                          iy%beg:iy%end, &
                                                          iz%beg:iz%end ))
                                    IF (n > 0) THEN
                                        ALLOCATE(dqL_prim_dy_ndqp(i,j,k)%vf( &
                                                       E_idx+crv_idx(l))%sf( &
                                                              ix%beg:ix%end, &
                                                              iy%beg:iy%end, &
                                                              iz%beg:iz%end ))
                                        ALLOCATE(dqR_prim_dy_ndqp(i,j,k)%vf( &
                                                       E_idx+crv_idx(l))%sf( &
                                                              ix%beg:ix%end, &
                                                              iy%beg:iy%end, &
                                                              iz%beg:iz%end ))
                                        IF (p > 0) THEN
                                            ALLOCATE(dqL_prim_dz_ndqp(i,j,k)%vf( &
                                                           E_idx+crv_idx(l))%sf( &
                                                                  ix%beg:ix%end, &
                                                                  iy%beg:iy%end, &
                                                                  iz%beg:iz%end ))
                                            ALLOCATE(dqR_prim_dz_ndqp(i,j,k)%vf( &
                                                           E_idx+crv_idx(l))%sf( &
                                                                  ix%beg:ix%end, &
                                                                  iy%beg:iy%end, &
                                                                  iz%beg:iz%end ))
                                        END IF
                                    END IF
                                END DO

                            END IF
                        
                        END DO
                    END DO
                END DO
            END IF
            ! END: Allocation/Association of dqK_prim_ds_ndqp ==================
            
            
            ! Allocation of gm_alpha_qp ========================================
            ALLOCATE(gm_alpha_qp( ieta%beg:ieta%end, &
                                  iksi%beg:iksi%end, &
                                  itau%beg:itau%end ))
            
            IF(We_size > 0) THEN
                DO k = itau%beg, itau%end
                    DO j = iksi%beg, iksi%end
                        DO i = ieta%beg, ieta%end
                        
                            ALLOCATE(gm_alpha_qp(i,j,k)%vf(1:num_fluids))
                        
                            IF(ANY(SUM(ABS((/i,j,k/))) == (/0,num_dims/))) THEN
                                DO l = 1, crv_size
                                    ALLOCATE(gm_alpha_qp(i,j,k)%vf(crv_idx(l))%sf( &
                                                                    ix%beg:ix%end, &
                                                                    iy%beg:iy%end, &
                                                                    iz%beg:iz%end ))
                                END DO
                            END IF
                        
                        END DO
                    END DO
                END DO
            END IF
            ! ==================================================================
            
            
            ! Allocation of gm_alphaK_ndqp =====================================
            ALLOCATE(gm_alphaL_ndqp(     1   :num_dims, &
                                     ichi%beg:ichi%end, &
                                     ipsi%beg:ipsi%end ))
            ALLOCATE(gm_alphaR_ndqp(     1   :num_dims, &
                                     ichi%beg:ichi%end, &
                                     ipsi%beg:ipsi%end ))
            
            IF(We_size > 0) THEN
                DO k = ipsi%beg, ipsi%end
                    DO j = ichi%beg, ichi%end
                        DO i = 1, num_dims
                        
                            ALLOCATE(gm_alphaL_ndqp(i,j,k)%vf(1:num_fluids))
                            ALLOCATE(gm_alphaR_ndqp(i,j,k)%vf(1:num_fluids))
                        
                            IF (ABS(j) >= ABS(k)) THEN
                                DO l = 1, crv_size
                                    ALLOCATE(gm_alphaL_ndqp(i,j,k)%vf( &
                                                       crv_idx(l))%sf( &
                                                        ix%beg:ix%end, &
                                                        iy%beg:iy%end, &
                                                        iz%beg:iz%end ))
                                    ALLOCATE(gm_alphaR_ndqp(i,j,k)%vf( &
                                                       crv_idx(l))%sf( &
                                                        ix%beg:ix%end, &
                                                        iy%beg:iy%end, &
                                                        iz%beg:iz%end ))
                                END DO
                            END IF
                        
                        END DO
                    END DO
                END DO
            END IF
            ! ==================================================================
            
            
            ! Allocation of un_alphaK_s_ndqp ===================================
            IF(We_size > 0) THEN
               
                ALLOCATE(un_alphaL_x_ndqp(     1   :num_dims, &
                                           ichi%beg:ichi%end, &
                                           ipsi%beg:ipsi%end ))
                ALLOCATE(un_alphaL_y_ndqp(     1   :num_dims, &
                                           ichi%beg:ichi%end, &
                                           ipsi%beg:ipsi%end ))
                ALLOCATE(un_alphaL_z_ndqp(     1   :num_dims, &
                                           ichi%beg:ichi%end, &
                                           ipsi%beg:ipsi%end ))
                ALLOCATE(un_alphaR_x_ndqp(     1   :num_dims, &
                                           ichi%beg:ichi%end, &
                                           ipsi%beg:ipsi%end ))
                ALLOCATE(un_alphaR_y_ndqp(     1   :num_dims, &
                                           ichi%beg:ichi%end, &
                                           ipsi%beg:ipsi%end ))
                ALLOCATE(un_alphaR_z_ndqp(     1   :num_dims, &
                                           ichi%beg:ichi%end, &
                                           ipsi%beg:ipsi%end ))
                
                DO k = ipsi%beg, ipsi%end
                    DO j = ichi%beg, ichi%end
                      
                        DO i = 1, num_dims
                            ALLOCATE(un_alphaL_x_ndqp(i,j,k)%vf(1:num_fluids))
                            ALLOCATE(un_alphaL_y_ndqp(i,j,k)%vf(1:num_fluids))
                            ALLOCATE(un_alphaL_z_ndqp(i,j,k)%vf(1:num_fluids))
                            ALLOCATE(un_alphaR_x_ndqp(i,j,k)%vf(1:num_fluids))
                            ALLOCATE(un_alphaR_y_ndqp(i,j,k)%vf(1:num_fluids))
                            ALLOCATE(un_alphaR_z_ndqp(i,j,k)%vf(1:num_fluids))
                        END DO
                      
                        IF (ABS(j) >= ABS(k)) THEN
            
                            DO l = 1, crv_size
                                ALLOCATE(un_alphaL_x_ndqp(1,j,k)%vf(crv_idx(l))%sf( &
                                                                     ix%beg:ix%end, &
                                                                     iy%beg:iy%end, &
                                                                     iz%beg:iz%end ))
                                ALLOCATE(un_alphaR_x_ndqp(1,j,k)%vf(crv_idx(l))%sf( &
                                                                     ix%beg:ix%end, &
                                                                     iy%beg:iy%end, &
                                                                     iz%beg:iz%end ))
                                IF (n > 0) THEN
                                    ALLOCATE(un_alphaL_y_ndqp(2,j,k)%vf(crv_idx(l))%sf( &
                                                                         ix%beg:ix%end, &
                                                                         iy%beg:iy%end, &
                                                                         iz%beg:iz%end ))
                                    ALLOCATE(un_alphaR_y_ndqp(2,j,k)%vf(crv_idx(l))%sf( &
                                                                         ix%beg:ix%end, &
                                                                         iy%beg:iy%end, &
                                                                         iz%beg:iz%end ))
                                    IF (p > 0) THEN
                                        ALLOCATE(un_alphaL_z_ndqp(3,j,k)%vf(crv_idx(l))%sf( &
                                                                             ix%beg:ix%end, &
                                                                             iy%beg:iy%end, &
                                                                             iz%beg:iz%end ))
                                        ALLOCATE(un_alphaR_z_ndqp(3,j,k)%vf(crv_idx(l))%sf( &
                                                                             ix%beg:ix%end, &
                                                                             iy%beg:iy%end, &
                                                                             iz%beg:iz%end ))
                                    END IF
                                END IF
                            END DO
                        END IF

                    END DO
                END DO
               
            END IF
            ! END: Allocation of un_alphaK_s_ndqp ==============================
            
            
            ! Allocation of kappa_vf ===========================================
            IF(We_size > 0) THEN
               
                ALLOCATE(kappa_vf(1:num_fluids))
               
                DO i = 1, crv_size
                    ALLOCATE(kappa_vf(crv_idx(i))%sf( ix%beg:ix%end, &
                                                      iy%beg:iy%end, &
                                                      iz%beg:iz%end ))
                END DO
               
            END IF
            ! ==================================================================
            
            
            ! Allocation/Association of kappaK_ndqp ============================
            ALLOCATE(kappaL_ndqp(     1   :num_dims, &
                                  ichi%beg:ichi%end, &
                                  ipsi%beg:ipsi%end ))
            ALLOCATE(kappaR_ndqp(     1   :num_dims, &
                                  ichi%beg:ichi%end, &
                                  ipsi%beg:ipsi%end ))
            
            IF(We_size > 0) THEN
                DO k = ipsi%beg, ipsi%end
                    DO j = ichi%beg, ichi%end
                        DO i = 1, num_dims
                        
                            ALLOCATE(kappaL_ndqp(i,j,k)%vf(1:num_fluids))
                            ALLOCATE(kappaR_ndqp(i,j,k)%vf(1:num_fluids))
                        
                            IF(ABS(j) >= ABS(k)) THEN
                                IF(i == 1) THEN
                                    DO l = 1, crv_size
                                        ALLOCATE(kappaL_ndqp(i,j,k)%vf( &
                                                        crv_idx(l))%sf( &
                                                         ix%beg:ix%end, &
                                                         iy%beg:iy%end, &
                                                         iz%beg:iz%end ))
                                        ALLOCATE(kappaR_ndqp(i,j,k)%vf( &
                                                        crv_idx(l))%sf( &
                                                         ix%beg:ix%end, &
                                                         iy%beg:iy%end, &
                                                         iz%beg:iz%end ))
                                    END DO
                                ELSE
                                    DO l = 1, crv_size
                                        kappaL_ndqp(i,j,k)%vf(crv_idx(l))%sf => &
                                        kappaL_ndqp(1,j,k)%vf(crv_idx(l))%sf
                                        kappaR_ndqp(i,j,k)%vf(crv_idx(l))%sf => &
                                        kappaR_ndqp(1,j,k)%vf(crv_idx(l))%sf
                                    END DO
                                END IF
                            END IF
                        
                        END DO
                    END DO
                END DO
            END IF
            ! ==================================================================


            IF (alt_crv) THEN
                ! Allocation of dgm_alpha_ds_qp & un_alpha_s_qp ================
                ALLOCATE(dgm_alpha_dx_qp(ieta%beg:ieta%end, &
                                         iksi%beg:iksi%end, &
                                         itau%beg:itau%end ))
                ALLOCATE(dgm_alpha_dy_qp(ieta%beg:ieta%end, &
                                         iksi%beg:iksi%end, &
                                         itau%beg:itau%end ))
                ALLOCATE(dgm_alpha_dz_qp(ieta%beg:ieta%end, &
                                         iksi%beg:iksi%end, &
                                         itau%beg:itau%end ))
                ALLOCATE(un_alpha_x_qp(ieta%beg:ieta%end, &
                                       iksi%beg:iksi%end, &
                                       itau%beg:itau%end ))
                ALLOCATE(un_alpha_y_qp(ieta%beg:ieta%end, &
                                       iksi%beg:iksi%end, &
                                       itau%beg:itau%end ))
                ALLOCATE(un_alpha_z_qp(ieta%beg:ieta%end, &
                                       iksi%beg:iksi%end, &
                                       itau%beg:itau%end ))
        
                IF (We_size > 0) THEN
                    DO k = itau%beg, itau%end
                        DO j = iksi%beg, iksi%end
                            DO i = ieta%beg, ieta%end
                                ALLOCATE(dgm_alpha_dx_qp(i,j,k)%vf(1:num_fluids))
                                ALLOCATE(dgm_alpha_dy_qp(i,j,k)%vf(1:num_fluids))
                                ALLOCATE(dgm_alpha_dz_qp(i,j,k)%vf(1:num_fluids))
                                ALLOCATE(un_alpha_x_qp(i,j,k)%vf(1:num_fluids))
                                ALLOCATE(un_alpha_y_qp(i,j,k)%vf(1:num_fluids))
                                ALLOCATE(un_alpha_z_qp(i,j,k)%vf(1:num_fluids))
        
                                IF (ANY(SUM(ABS((/i,j,k/))) == (/0,num_dims/))) THEN
                                    DO l = 1, crv_size
                                        ALLOCATE(dgm_alpha_dx_qp(i,j,k)%vf(crv_idx(l))%sf( &
                                                                            ix%beg:ix%end, &
                                                                            iy%beg:iy%end, &
                                                                            iz%beg:iz%end ))
                                        ALLOCATE(dgm_alpha_dy_qp(i,j,k)%vf(crv_idx(l))%sf( &
                                                                            ix%beg:ix%end, &
                                                                            iy%beg:iy%end, &
                                                                            iz%beg:iz%end ))
                                        ALLOCATE(dgm_alpha_dz_qp(i,j,k)%vf(crv_idx(l))%sf( &
                                                                            ix%beg:ix%end, &
                                                                            iy%beg:iy%end, &
                                                                            iz%beg:iz%end ))
                                        ALLOCATE(un_alpha_x_qp(i,j,k)%vf(crv_idx(l))%sf( &
                                                                          ix%beg:ix%end, &
                                                                          iy%beg:iy%end, &
                                                                          iz%beg:iz%end ))
                                        ALLOCATE(un_alpha_y_qp(i,j,k)%vf(crv_idx(l))%sf( &
                                                                          ix%beg:ix%end, &
                                                                          iy%beg:iy%end, &
                                                                          iz%beg:iz%end ))
                                        ALLOCATE(un_alpha_z_qp(i,j,k)%vf(crv_idx(l))%sf( &
                                                                          ix%beg:ix%end, &
                                                                          iy%beg:iy%end, &
                                                                          iz%beg:iz%end ))
                                    END DO
                                END IF
        
                            END DO
                        END DO
                    END DO
                END IF
            END IF
            ! ==============================================================
    
    
            ! Allocation of laplacian_vf and alt_kappa_vf
            IF (We_size > 0) THEN
                IF (alt_crv) THEN
                    ALLOCATE(laplacian_vf(1:num_fluids))
                    ALLOCATE(alt_kappa_vf(1:num_fluids))

                    DO i = 1, crv_size
                        ALLOCATE(laplacian_vf(crv_idx(i))%sf( ix%beg:ix%end, &
                                                              iy%beg:iy%end, &
                                                              iz%beg:iz%end ))
                        ALLOCATE(alt_kappa_vf(crv_idx(i))%sf( ix%beg:ix%end, &
                                                              iy%beg:iy%end, &
                                                              iz%beg:iz%end ))
                    END DO
                END IF
                
                IF (We_src) THEN
                    ALLOCATE(We_mtm_src(0:m,0:n,0:p))
                    ALLOCATE(We_nrg_src(0:m,0:n,0:p))
                END IF

            END IF
    
            ! Allocation of regularization terms
            IF (regularization) THEN
                ALLOCATE(reg_src_vf(1:sys_size))
                DO i = 1, sys_size
                    ALLOCATE(reg_src_vf(i)%sf(0:m,0:n,0:p))
                END DO
            END IF

            if (bubbles) then
                allocate( bub_adv_src(0:m,0:n,0:p) )
                allocate( bub_r_src(1:nb,0:m,0:n,0:p) )
                allocate( bub_v_src(1:nb,0:m,0:n,0:p) )
                allocate( bub_p_src(1:nb,0:m,0:n,0:p) )
                allocate( bub_m_src(1:nb,0:m,0:n,0:p) )
            end if

            if (monopole) then
                allocate( mono_mass_src(0:m,0:n,0:p) )
                allocate( mono_mom_src(1:num_dims,0:m,0:n,0:p) )
                allocate( mono_E_src(0:m,0:n,0:p) )
            end if

            allocate( divu%sf( &
                                                        ix%beg:ix%end, &
                                                        iy%beg:iy%end, &
                                                        iz%beg:iz%end ))
 
            ! Configuring Coordinate Direction Indexes =========================
            ix%beg = -1; IF(n > 0) iy%beg = -1; IF(p > 0) iz%beg = -1
            
            ix%end = m; iy%end = n; iz%end = p
            ! ==================================================================
            
            ! Allocation/Association of flux_ndqp, flux_src_ndqp, and flux_gsrc_ndqp ===
            ALLOCATE(    flux_ndqp(     1   :num_dims, &
                                    ichi%beg:ichi%end, &
                                    ipsi%beg:ipsi%end ))
            ALLOCATE(flux_src_ndqp(     1   :num_dims, &
                                    ichi%beg:ichi%end, &
                                    ipsi%beg:ipsi%end ))
            ALLOCATE(flux_gsrc_ndqp(     1   :num_dims, &
                                    ichi%beg:ichi%end, &
                                    ipsi%beg:ipsi%end ))
            
            ! Allocation of variables for flux limiter
            IF (tvd_rhs_flux) THEN
                ALLOCATE(flux_lim_func(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
    
                ALLOCATE(    lo_flux_ndqp(  1   :num_dims, &
                            ichi%beg:ichi%end, &
                            ipsi%beg:ipsi%end ))
                ALLOCATE(lo_flux_src_ndqp(  1   :num_dims, &
                            ichi%beg:ichi%end, &
                            ipsi%beg:ipsi%end ))
                ALLOCATE(lo_flux_gsrc_ndqp(  1   :num_dims, &
                            ichi%beg:ichi%end, &
                            ipsi%beg:ipsi%end ))
                ALLOCATE(    hi_flux_ndqp(  1   :num_dims, &
                            ichi%beg:ichi%end, &
                            ipsi%beg:ipsi%end ))
                ALLOCATE(hi_flux_src_ndqp(  1   :num_dims, &
                            ichi%beg:ichi%end, &
                            ipsi%beg:ipsi%end ))
                ALLOCATE(hi_flux_gsrc_ndqp(  1   :num_dims, &
                            ichi%beg:ichi%end, &
                            ipsi%beg:ipsi%end ))
            END IF

            DO k = ipsi%beg, ipsi%end
                DO j = ichi%beg, ichi%end
                    DO i = 1, num_dims
                     
                        ALLOCATE(    flux_ndqp(i,j,k)%vf(1:sys_size))
                        ALLOCATE(flux_src_ndqp(i,j,k)%vf(1:sys_size))
                        ALLOCATE(flux_gsrc_ndqp(i,j,k)%vf(1:sys_size))
                        
                        IF(ABS(j) >= ABS(k)) THEN
                            IF(i == 1) THEN
                              
                                DO l = 1, sys_size
                                    ALLOCATE(flux_ndqp(i,j,k)%vf(l)%sf( &
                                                         ix%beg:ix%end, &
                                                         iy%beg:iy%end, &
                                                         iz%beg:iz%end ))
                                    ALLOCATE(flux_gsrc_ndqp(i,j,k)%vf(l)%sf( &
                                                              ix%beg:ix%end, &
                                                              iy%beg:iy%end, &
                                                              iz%beg:iz%end ))
                                END DO
                              
                                IF(ANY(Re_size > 0) .OR. We_size > 0) THEN
                                    DO l = mom_idx%beg, E_idx
                                        ALLOCATE(flux_src_ndqp(i,j,k)%vf(l)%sf(     &
                                                                 ix%beg:ix%end,     &
                                                                 iy%beg:iy%end,     &
                                                                 iz%beg:iz%end ))
                                    END DO
                                END IF
                            
                                !SHB: Edit for 2 fluids w/ bubbles
                                !IF (bubbles .and. riemann_solver .ne. 1) then
                                !    DO l = adv_idx%beg,adv_idx%end
                                !        ALLOCATE(flux_src_ndqp(i,j,k)%vf(l)%sf(       &
                                !                                   ix%beg:ix%end,   &
                                !                                   iy%beg:iy%end,   &
                                !                                   iz%beg:iz%end ))
                                !    END DO
                                !END IF
                                ALLOCATE(flux_src_ndqp(i,j,k)%vf(adv_idx%beg)%sf(       &
                                                                   ix%beg:ix%end,   &
                                                                   iy%beg:iy%end,   &
                                                                   iz%beg:iz%end ))
                                IF(riemann_solver == 1) THEN
                                    DO l = adv_idx%beg+1, adv_idx%end
                                        ALLOCATE(flux_src_ndqp(i,j,k)%vf(l)%sf( &
                                                                 ix%beg:ix%end, &
                                                                 iy%beg:iy%end, &
                                                                 iz%beg:iz%end ))
                                    END DO
                                ELSE
                                    !if ( (num_fluids > 1) .and. (bubbles .neqv. .TRUE.)) then
                                        DO l = adv_idx%beg+1, adv_idx%end
                                            flux_src_ndqp(i,j,k)%vf(l)%sf => &
                                            flux_src_ndqp(i,j,k)%vf(adv_idx%beg)%sf
                                        END DO
                                    !end if
                                END IF
                              
                            ELSE
                              
                                DO l = 1, sys_size
                                        flux_ndqp(i,j,k)%vf(l)%sf => &
                                        flux_ndqp(1,j,k)%vf(l)%sf
                                    flux_src_ndqp(i,j,k)%vf(l)%sf => &
                                    flux_src_ndqp(1,j,k)%vf(l)%sf
                                    flux_gsrc_ndqp(i,j,k)%vf(l)%sf => &
                                    flux_gsrc_ndqp(1,j,k)%vf(l)%sf
                                END DO
                              
                            END IF
                        END IF
                     
                    END DO
                END DO
            END DO

            IF (tvd_rhs_flux) THEN
                DO k = ipsi%beg, ipsi%end
                    DO j = ichi%beg, ichi%end
                        DO i = 1, num_dims
                        
                            ALLOCATE(    lo_flux_ndqp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(lo_flux_src_ndqp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(lo_flux_gsrc_ndqp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(    hi_flux_ndqp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(hi_flux_src_ndqp(i,j,k)%vf(1:sys_size))
                            ALLOCATE(hi_flux_gsrc_ndqp(i,j,k)%vf(1:sys_size))
    
                            IF (ABS(j) >= ABS(k)) THEN
                                IF (i == 1) THEN
                                    DO l = 1, sys_size
                                        ALLOCATE(lo_flux_ndqp(i,j,k)%vf(l)%sf( &
                                                                ix%beg:ix%end, &
                                                                iy%beg:iy%end, &
                                                                iz%beg:iz%end ))
                                        ALLOCATE(hi_flux_ndqp(i,j,k)%vf(l)%sf( &
                                                                ix%beg:ix%end, &
                                                                iy%beg:iy%end, &
                                                                iz%beg:iz%end ))
                                        ALLOCATE(lo_flux_gsrc_ndqp(i,j,k)%vf(l)%sf( &
                                                                     ix%beg:ix%end, &
                                                                     iy%beg:iy%end, &
                                                                     iz%beg:iz%end ))
                                        ALLOCATE(hi_flux_gsrc_ndqp(i,j,k)%vf(l)%sf( &
                                                                     ix%beg:ix%end, &
                                                                     iy%beg:iy%end, &
                                                                     iz%beg:iz%end ))
                                    END DO
    
                                    IF (ANY(Re_size > 0) .OR. We_size > 0) THEN
                                        DO l = mom_idx%beg, E_idx
                                            ALLOCATE(lo_flux_src_ndqp(i,j,k)%vf(l)%sf( &
                                                                        ix%beg:ix%end, &
                                                                        iy%beg:iy%end, &
                                                                        iz%beg:iz%end ))
                                            ALLOCATE(hi_flux_src_ndqp(i,j,k)%vf(l)%sf( &
                                                                        ix%beg:ix%end, &
                                                                        iy%beg:iy%end, &
                                                                        iz%beg:iz%end ))
                                        END DO
                                    END IF
    
                                    ALLOCATE(lo_flux_src_ndqp(i,j,k)%vf(adv_idx%beg)%sf( &
                                                                          ix%beg:ix%end, &
                                                                          iy%beg:iy%end, &
                                                                          iz%beg:iz%end ))
                                    ALLOCATE(hi_flux_src_ndqp(i,j,k)%vf(adv_idx%beg)%sf( &
                                                                          ix%beg:ix%end, &
                                                                          iy%beg:iy%end, &
                                                                          iz%beg:iz%end ))
                                    IF (riemann_solver == 1) THEN
                                        DO l = adv_idx%beg+1, adv_idx%end
                                            ALLOCATE(lo_flux_src_ndqp(i,j,k)%vf(l)%sf( &
                                                                        ix%beg:ix%end, &
                                                                        iy%beg:iy%end, &
                                                                        iz%beg:iz%end ))
                                            ALLOCATE(hi_flux_src_ndqp(i,j,k)%vf(l)%sf( &
                                                                        ix%beg:ix%end, &
                                                                        iy%beg:iy%end, &
                                                                        iz%beg:iz%end ))
                                        END DO
                                    ELSE
                                        DO l = adv_idx%beg+1, adv_idx%end
                                            lo_flux_src_ndqp(i,j,k)%vf(l)%sf => &
                                            lo_flux_src_ndqp(i,j,k)%vf(adv_idx%beg)%sf
                                            hi_flux_src_ndqp(i,j,k)%vf(l)%sf => &
                                            hi_flux_src_ndqp(i,j,k)%vf(adv_idx%beg)%sf
                                        END DO
                                    END IF
                                ELSE
                                    DO l = 1, sys_size
                                            lo_flux_ndqp(i,j,k)%vf(l)%sf => &
                                            lo_flux_ndqp(1,j,k)%vf(l)%sf
                                            hi_flux_ndqp(i,j,k)%vf(l)%sf => &
                                            hi_flux_ndqp(1,j,k)%vf(l)%sf
                                        lo_flux_src_ndqp(i,j,k)%vf(l)%sf => &
                                        lo_flux_src_ndqp(1,j,k)%vf(l)%sf
                                        hi_flux_src_ndqp(i,j,k)%vf(l)%sf => &
                                        hi_flux_src_ndqp(1,j,k)%vf(l)%sf
                                       lo_flux_gsrc_ndqp(i,j,k)%vf(l)%sf => &
                                       lo_flux_gsrc_ndqp(1,j,k)%vf(l)%sf
                                       hi_flux_gsrc_ndqp(i,j,k)%vf(l)%sf => &
                                       hi_flux_gsrc_ndqp(1,j,k)%vf(l)%sf
                                    END DO
                                END IF
                            END IF
                        END DO
                    END DO
                END DO
            END IF
            ! END: Allocation/Association of flux_ndqp, flux_src_ndqp, and flux_gsrc_ndqp ===
            
            
            ! Associating procedural pointer to the subroutine that will be
            ! utilized to calculate the solution of a given Riemann problem
            IF(riemann_solver == 1) THEN
                s_riemann_solver => s_hll_riemann_solver
            ELSEIF(riemann_solver == 2) THEN
                s_riemann_solver => s_hllc_riemann_solver
            ELSE
                s_riemann_solver => s_exact_riemann_solver
            END IF
            
            ! Associating the procedural pointer to the appropriate subroutine
            ! that will be utilized in the conversion to the mixture variables
            IF (model_eqns == 1) THEN        ! Gamma/pi_inf model
                s_convert_to_mixture_variables => &
                             s_convert_mixture_to_mixture_variables
            ELSE IF (bubbles) THEN        !SHB volume fraction for bubbles
                s_convert_to_mixture_variables => &
                             s_convert_species_to_mixture_variables_bubbles 
            ELSE                            ! Volume fraction model
                s_convert_to_mixture_variables => &
                             s_convert_species_to_mixture_variables
            END IF
            
        END SUBROUTINE s_initialize_rhs_module ! -------------------------------
        
        
        
        
        
        SUBROUTINE s_compute_rhs(q_cons_vf, q_prim_vf, rhs_vf, t_step) ! -------
        ! Description: The purpose of this procedure is to employ the inputted
        !              cell-average conservative variables in order to compute
        !              the cell-average RHS variables of the semidiscrete form
        !              of the governing equations by utilizing the appropriate
        !              Riemann solver.
            
            
            ! Cell-average conservative variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: q_cons_vf
            
            ! Cell-average primitive variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: q_prim_vf
            
            ! Cell-average RHS variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: rhs_vf

            ! Numerator and denominator when evaluating flux limiter function
            REAL(KIND(0d0)) :: top, bottom
            REAL(KIND(0d0)), DIMENSION(0:m,0:n,0:p) :: blkmod1, blkmod2, alpha1, alpha2, Kterm
            
            ! Current time-step
            INTEGER, INTENT(IN) :: t_step
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l,r
            
            
            ! Configuring Coordinate Direction Indexes =========================
            ix%beg = -buff_size; iy%beg = 0; iz%beg = 0
            
            IF(n > 0) iy%beg = -buff_size; IF(p > 0) iz%beg = -buff_size
            
            ix%end = m - ix%beg; iy%end = n - iy%beg; iz%end = p - iz%beg
            ! ==================================================================
          
            
            ! Association/Population of Working Variables ======================
            DO i = 1, sys_size
                q_cons_qp(0,0,0)%vf(i)%sf => q_cons_vf(i)%sf
                q_prim_qp(0,0,0)%vf(i)%sf => q_prim_vf(i)%sf
            END DO
            
            CALL s_populate_conservative_variables_buffers()
           
            ! does this, same loop problem as below?
            IF((model_eqns == 2 .OR. model_eqns == 3) .AND. (adv_alphan .NEQV. .TRUE.)) THEN
                !was a modified line; sys_size -> adv_idx%end
                !q_cons_qp(0,0,0)%vf(adv_idx%end)%sf = 1d0
                
                !DO i = adv_idx%beg, adv_idx%end
                !    q_cons_qp(0,0,0)%vf(adv_idx%end)%sf = &
                !    q_cons_qp(0,0,0)%vf(adv_idx%end)%sf - &
                !    q_cons_qp(0,0,0)%vf(i)%sf
                !END DO
 
                !from old code
                q_cons_qp(0,0,0)%vf(sys_size)%sf = 1d0
                
                DO i = adv_idx%beg, adv_idx%end
                    q_cons_qp(0,0,0)%vf(sys_size)%sf = &
                    q_cons_qp(0,0,0)%vf(sys_size)%sf - &
                    q_cons_qp(0,0,0)%vf(i)%sf
                END DO
            END IF

            IF (mpp_lim .and. bubbles) THEN
                !adjust volume fractions, according to modeled gas void fraction
                alf_sum%sf = 0d0
                DO i = adv_idx%beg, adv_idx%end - 1
                    alf_sum%sf = alf_sum%sf + q_cons_vf(i)%sf
                END DO

                DO i = adv_idx%beg, adv_idx%end-1
                    q_cons_vf(i)%sf = q_cons_vf(i)%sf * (1.d0 - q_cons_vf(alf_idx)%sf) &
                            / alf_sum%sf
                END DO
            END IF

            !! Correct void fraction if negative
            !if (bubbles) then
            !    do r = 1,nb; do i = 0,m; do j = 0,n; do k = 0,p
            !        if( (q_cons_vf(alf_idx)%sf(i,j,k) < 0d0)        .or. &
            !            (q_cons_vf(bub_idx%rs(r))%sf(i,j,k)) < 0d0       &
            !            ) then
            !            q_cons_vf(alf_idx)%sf(i,j,k) = 1.d-12
            !            q_cons_vf(bub_idx%rs(r))%sf(i,j,k) = 1.d-12
            !            q_cons_vf(bub_idx%vs(r))%sf(i,j,k) = 1.d-12
            !        end if
            !    end do; end do; end do; end do
            !end if

            do j = 1,sys_size
                print*, sum( q_cons_vf(j)%sf(0:m,0:n,0:p))
            end do
            ! ==================================================================


            ! Computing Volume Fraction Gradients and Curvatures ===============
            IF (We_size > 0 .AND. lsq_deriv) THEN
                IF (p > 0) THEN
                    CALL s_compute_lsq_gradient_curvature(  q_prim_qp(0,0,0)%vf, &
                                                        dq_prim_dx_qp(0,0,0)%vf, &
                                                        dq_prim_dy_qp(0,0,0)%vf, &
                                                        dq_prim_dz_qp(0,0,0)%vf, &
                                                          gm_alpha_qp(0,0,0)%vf, &
                                                          kappa_vf)
                ELSE
                    CALL s_compute_lsq_gradient_curvature(  q_prim_qp(0,0,0)%vf, &
                                                        dq_prim_dx_qp(0,0,0)%vf, &
                                                        dq_prim_dy_qp(0,0,0)%vf, &
                                                        dq_prim_dy_qp(0,0,0)%vf, &
                                                          gm_alpha_qp(0,0,0)%vf, &
                                                          kappa_vf)
                END IF
            END IF
  
            if (crv_size > 0) call s_get_crv(q_cons_vf,q_prim_vf,rhs_vf)

            ! Converting Conservative to Primitive Variables ===================
            iv%beg = 1; iv%end = adv_idx%end
            
            !won't do anything without flag turned on
            CALL s_reconstruct_cell_interior_values(q_cons_qp)
           
            !won't do anything without commute_err
            IF( (model_eqns == 2 .OR. model_eqns == 3)  &
                           .AND.           &
                (adv_alphan .NEQV. .TRUE.) &
                           .AND.           &
                        commute_err        ) THEN
                DO k = itau%beg, itau%end, 2
                    DO j = iksi%beg, iksi%end, 2
                        DO i = ieta%beg, ieta%end, 2
                            
                            !was a modified line, don't use anyway
                            q_cons_qp(i,j,k)%vf(adv_idx%end)%sf = 1d0
                        
                            DO l = adv_idx%beg, adv_idx%end
                                q_cons_qp(i,j,k)%vf(adv_idx%end)%sf = &
                                q_cons_qp(i,j,k)%vf(adv_idx%end)%sf - &
                                q_cons_qp(i,j,k)%vf(l)%sf
                            END DO
                        
                        END DO
                    END DO
                END DO
            END IF

            !convert conservative variables to primative 
            !   (except first and last, \alpha \rho and \alpha)
            !we do this, though there is no actual loop, just (0,0,0)
            DO k = itau%beg, itau%end, 2
                DO j = iksi%beg, iksi%end, 2
                    DO i = ieta%beg, ieta%end, 2
                        CALL s_convert_conservative_to_primitive_variables( &
                                                       q_cons_qp(i,j,k)%vf, &
                                                       q_prim_qp(i,j,k)%vf, &
                                                     gm_alpha_qp(i,j,k)%vf, &
                                                                  ix,iy,iz  )
                    END DO
                END DO
            END DO

            !get the coeffients for splitting gas and bubble terms
            !if (bubbles .and. (num_fluids > 1)) then
            if ( bubbles ) then
                do i = ix%beg,ix%end; do j = iy%beg,iy%end; do k = iz%beg,iz%end
                    if (num_fluids .le. 2) then
                        call s_get_coeffs(a_coeff(i,j,k),b_coeff(i,j,k),q_cons_qp(0,0,0)%vf(alf_idx)%sf(i,j,k))
                    else if (num_fluids == 3) then
                        call s_get_scoeffs( &
                            a_coeff(i,j,k), &
                            b_coeff(i,j,k), &
                            s_term(i,j,k), &
                            q_cons_qp(0,0,0)%vf(alf_idx-1)%sf(i,j,k), &
                            q_cons_qp(0,0,0)%vf(alf_idx)%sf(i,j,k) )
                    end if
                end do; end do; end do
                if (proc_rank == 0 .and. bubble_switch) then
                    print*, 'a/b coeffs',maxval(abs(q_cons_qp(0,0,0)%vf(alf_idx)%sf)),maxval(abs(a_coeff)), maxval(abs(b_coeff))
                    do i = adv_idx%beg,adv_idx%end
                        print*, 'max alfs',maxval(abs(q_cons_qp(0,0,0)%vf(i)%sf))
                    end do
                end if
            end if

            iv%beg = mom_idx%beg; iv%end = E_idx !won't use these SHB
           
            !won't do anything without flag turned on
            CALL s_average_cell_interior_values(q_prim_qp)
           
            IF(t_step == t_step_stop) RETURN
            ! ==================================================================
            
            ! Computing Velocity Gradients =====================================
            if (any(Re_size > 0)) call s_get_viscous(q_cons_vf,q_prim_vf,rhs_vf)

            PRINT '(A)', 'SHB: Before dim splitting'
            
            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ! Dimensional Splitting Loop =======================================
            DO i = 1, num_dims
               
                ! Configuring Coordinate Direction Indexes ======================
                ix%beg = -buff_size; iy%beg = 0; iz%beg = 0
                
                IF(n > 0) iy%beg = -buff_size; IF(p > 0) iz%beg = -buff_size
                
                ix%end = m - ix%beg; iy%end = n - iy%beg; iz%end = p - iz%beg
                ! ===============================================================
                
                ! we do all this stuff
                ! Reconstructing Primitive/Conservative Variables ===============
                IF(char_decomp .OR. (ALL(Re_size == 0) .AND. We_size == 0)) THEN
                   
                    !IF(char_decomp) THEN
                    !    iv%beg = 1; iv%end = sys_size
                    !ELSE
                    !    iv%beg = 1; iv%end = adv_idx%end
                    !END IF
                    !iv%beg = 1; iv%end = adv_idx%end
                    iv%beg = 1;
                    if (adv_alphan) then
                        iv%end = adv_idx%end !old  
                        if (bubbles) iv%end = sys_size  !SHB: modified line to look out for
                    else
                        iv%end = adv_idx%end+1
                    end if
                        
                    !reconstruct either primative or conservative vars 
                    IF(weno_vars == 1) THEN
                        CALL s_reconstruct_cell_boundary_values(       &
                                   q_cons_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                  qL_cons_ndqp(i,:,:), &
                                                  qR_cons_ndqp(i,:,:), &
                                                         weno_vars, i  )
                    ELSE
                        CALL s_reconstruct_cell_boundary_values(       &
                                   q_prim_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                  qL_prim_ndqp(i,:,:), &
                                                  qR_prim_ndqp(i,:,:), &
                                                         weno_vars, i  )
                    END IF

                ELSE
                !don't do this stuff, just reconstruct all primative variables at once
                ! ===============================================================
               
                ! Reconstructing Continuity Variables ===========================
                    IF(weno_vars == 2 .OR. ALL(Re_size == 0)) THEN
                     
                        iv%beg = cont_idx%beg; iv%end = cont_idx%end
                     
                        CALL s_reconstruct_cell_boundary_values(       &
                                   q_cons_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                  qL_cons_ndqp(i,:,:), &
                                                  qR_cons_ndqp(i,:,:), &
                                                          dflt_int, i  )
                     
                    END IF
                ! ===============================================================
               
               ! Reconstructing Momentum/Velocity Variables ====================
                    IF(ALL(Re_size == 0)) THEN
                     
                        iv%beg = mom_idx%beg; iv%end = mom_idx%end
                     
                        IF(weno_vars == 1) THEN
                            CALL s_reconstruct_cell_boundary_values(      &
                                      q_cons_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                     qL_cons_ndqp(i,:,:), &
                                                     qR_cons_ndqp(i,:,:), &
                                                             dflt_int, i  )
                        ELSE
                            CALL s_reconstruct_cell_boundary_values(      &
                                      q_prim_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                     qL_prim_ndqp(i,:,:), &
                                                     qR_prim_ndqp(i,:,:), &
                                                             dflt_int, i  )
                        END IF
                     
                    END IF
               ! ===============================================================
               
               ! Reconstructing Partial or Mixture Energy/Pressure Variables ===
                    iv%beg = E_idx; iv%end = iv%beg
                  
                    IF(weno_vars == 1) THEN
                        CALL s_reconstruct_cell_boundary_values(       &
                                   q_cons_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                  qL_cons_ndqp(i,:,:), &
                                                  qR_cons_ndqp(i,:,:), &
                                                          dflt_int, i  )
                    ELSE
                        CALL s_reconstruct_cell_boundary_values(       &
                                   q_prim_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                  qL_prim_ndqp(i,:,:), &
                                                  qR_prim_ndqp(i,:,:), &
                                                          dflt_int, i  )
                    END IF
               ! ===============================================================
              
               ! Reconstructing Volume Fraction Variables ======================
                    IF(We_size == 0) THEN
                     
                        iv%beg = adv_idx%beg; iv%end = adv_idx%end
                     
                        CALL s_reconstruct_cell_boundary_values(       &
                                   q_cons_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                  qL_cons_ndqp(i,:,:), &
                                                  qR_cons_ndqp(i,:,:), &
                                                          dflt_int, i  )
                     
                    ELSE
                     
                        DO l = adv_idx%beg, adv_idx%end
                            IF(ALL(crv_idx /= l - E_idx)) THEN
                           
                                iv%beg = l; iv%end = iv%beg
                                
                                CALL s_reconstruct_cell_boundary_values(    &
                                        q_cons_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                       qL_cons_ndqp(i,:,:), &
                                                       qR_cons_ndqp(i,:,:), &
                                                               dflt_int, i  )
                           
                            END IF
                        END DO
                     
                    END IF
               ! ===============================================================

                END IF

                !should set \alpha_2 = 1 - \alpha_1
                IF((model_eqns == 2 .OR. model_eqns == 3) .AND. (adv_alphan .NEQV. .TRUE.)) THEN
                    DO k = ipsi%beg, ipsi%end, 2
                        DO j = ichi%beg, ichi%end, 2
                        
                            !from kevin
                            !qL_cons_ndqp(i,j,k)%vf(adv_idx%end)%sf = 1d0
                            !qR_cons_ndqp(i,j,k)%vf(adv_idx%end)%sf = 1d0
                            
                            !DO l = adv_idx%beg, adv_idx%end
                               
                            !    qL_cons_ndqp(i,j,k)%vf(adv_idx%end)%sf = &
                            !    qL_cons_ndqp(i,j,k)%vf(adv_idx%end)%sf - &
                            !    qL_cons_ndqp(i,j,k)%vf(l)%sf !SHB: mistake?
                                
                            !    qR_cons_ndqp(i,j,k)%vf(adv_idx%end)%sf = &
                            !    qR_cons_ndqp(i,j,k)%vf(adv_idx%end)%sf - &
                            !    qR_cons_ndqp(i,j,k)%vf(l)%sf
                               
                            !END DO

                            !from old code
                            qL_cons_ndqp(i,j,k)%vf(sys_size)%sf = 1d0
                            qR_cons_ndqp(i,j,k)%vf(sys_size)%sf = 1d0
                            
                            DO l = adv_idx%beg, adv_idx%end
                               
                                qL_cons_ndqp(i,j,k)%vf(sys_size)%sf = &
                                qL_cons_ndqp(i,j,k)%vf(sys_size)%sf - &
                                qL_cons_ndqp(i,j,k)%vf(l)%sf !SHB: mistake?
                                
                                qR_cons_ndqp(i,j,k)%vf(sys_size)%sf = &
                                qR_cons_ndqp(i,j,k)%vf(sys_size)%sf - &
                                qR_cons_ndqp(i,j,k)%vf(l)%sf
                            END DO
                        
                        END DO
                    END DO
                END IF
               ! END: Reconstructing Volume Fraction Variables =================
               
               ! don't do, weno_vars == 2
               ! Converting Conservative to Primitive Variables ================
                IF(weno_vars == 1) THEN
                    DO k = ipsi%beg, ipsi%end, 2
                        DO j = ichi%beg, ichi%end, 2
                            CALL s_convert_conservative_to_primitive_variables( &
                                                        qL_cons_ndqp(i,j,k)%vf, &
                                                        qL_prim_ndqp(i,j,k)%vf, &
                                                      gm_alphaL_ndqp(i,j,k)%vf, &
                                                                      ix,iy,iz  )
                            CALL s_convert_conservative_to_primitive_variables( &
                                                        qR_cons_ndqp(i,j,k)%vf, &
                                                        qR_prim_ndqp(i,j,k)%vf, &
                                                      gm_alphaR_ndqp(i,j,k)%vf, &
                                                                      ix,iy,iz  )
                        END DO
                    END DO
                END IF
                ! ===============================================================
                
                ! Reconstructing First-Order Spatial Derivatives of Velocity ====
                IF(ANY(Re_size > 0)) THEN

                    iv%beg = mom_idx%beg; iv%end = mom_idx%end

                    IF (weno_Re_flux) THEN

                        CALL s_reconstruct_cell_boundary_values(         &
                                 dq_prim_dx_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                dqL_prim_dx_ndqp(i,:,:), &
                                                dqR_prim_dx_ndqp(i,:,:), &
                                                            dflt_int, i  )
                   
                        IF(n > 0) THEN
                           
                            CALL s_reconstruct_cell_boundary_values(         &
                                     dq_prim_dy_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                    dqL_prim_dy_ndqp(i,:,:), &
                                                    dqR_prim_dy_ndqp(i,:,:), &
                                                                dflt_int, i  )
                            IF(p > 0) THEN
                                CALL s_reconstruct_cell_boundary_values(         &
                                         dq_prim_dz_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                        dqL_prim_dz_ndqp(i,:,:), &
                                                        dqR_prim_dz_ndqp(i,:,:), &
                                                                   dflt_int, i  )
                            END IF

                        END IF

                    END IF
                   
                END IF
               ! ===============================================================
               
               ! Reconstructing Curvatures =====================================
                DO l = 1, crv_size
                  
                    iv%beg = crv_idx(l); iv%end = iv%beg

                    IF (alt_crv .NEQV. .TRUE.) THEN
                              CALL s_reconstruct_cell_boundary_values(   &
                                                kappa_vf(iv%beg:iv%end), &
                                                     kappaL_ndqp(i,:,:), &
                                                     kappaR_ndqp(i,:,:), &
                                                            dflt_int, i  )
                    ELSE
                        CALL s_reconstruct_cell_boundary_values(    &
                                    alt_kappa_vf(iv%beg:iv%end),    &
                                                kappaL_ndqp(i,:,:), &
                                                kappaR_ndqp(i,:,:), &
                                                        dflt_int, i )
                    END IF
                  
                END DO
               ! ===============================================================
               
               ! Configuring Coordinate Direction Indexes ======================
                IF(i == 1) THEN
                    ix%beg = -1; iy%beg =  0; iz%beg =  0
                ELSEIF(i == 2) THEN
                    ix%beg =  0; iy%beg = -1; iz%beg =  0
                ELSE
                    ix%beg =  0; iy%beg =  0; iz%beg = -1
                END IF
                
                ix%end = m; iy%end = n; iz%end = p
               ! ===============================================================
                print '(a)', 'SHB: before riemann solver flux and sources.'


                ! don't use
                IF (tvd_rhs_flux) THEN
                    ! Compute Riemann solver flux and source flux 
                    DO k = ipsi%beg, ipsi%end, 2
                        DO j = ichi%beg, ichi%end, 2
                            CALL s_riemann_solver(      qR_prim_ndqp(i,j,k)%vf, &
                                                    dqR_prim_dx_ndqp(i,j,k)%vf, &
                                                    dqR_prim_dy_ndqp(i,j,k)%vf, &
                                                    dqR_prim_dz_ndqp(i,j,k)%vf, &
                                                      gm_alphaR_ndqp(i,j,k)%vf, &
                                                         kappaR_ndqp(i,j,k)%vf, &
                                                        qL_prim_ndqp(i,j,k)%vf, &
                                                    dqL_prim_dx_ndqp(i,j,k)%vf, &
                                                    dqL_prim_dy_ndqp(i,j,k)%vf, &
                                                    dqL_prim_dz_ndqp(i,j,k)%vf, &
                                                      gm_alphaL_ndqp(i,j,k)%vf, &
                                                         kappaL_ndqp(i,j,k)%vf, &
                                                           q_prim_qp(0,0,0)%vf, &
                                                        hi_flux_ndqp(i,j,k)%vf, &
                                                    hi_flux_src_ndqp(i,j,k)%vf, &
                                                   hi_flux_gsrc_ndqp(i,j,k)%vf, &
                                                                    i, ix,iy,iz )
                        END DO
                    END DO
            
                    DO l = itau%beg, itau%end, 2
                        DO k = iksi%beg, iksi%end, 2
                            DO j = ieta%beg, ieta%end, 2
                                CALL s_riemann_solver(      q_prim_qp(j,k,l)%vf, &
                                                        dq_prim_dx_qp(j,k,l)%vf, &
                                                        dq_prim_dy_qp(j,k,l)%vf, &
                                                        dq_prim_dz_qp(j,k,l)%vf, &
                                                          gm_alpha_qp(j,k,l)%vf, &
                                                                       kappa_vf, &
                                                            q_prim_qp(j,k,l)%vf, &
                                                        dq_prim_dx_qp(j,k,l)%vf, &
                                                        dq_prim_dy_qp(j,k,l)%vf, &
                                                        dq_prim_dz_qp(j,k,l)%vf, &
                                                          gm_alpha_qp(j,k,l)%vf, &
                                                                       kappa_vf, &
                                                            q_prim_qp(0,0,0)%vf, &
                                                         lo_flux_ndqp(i,k,l)%vf, &
                                                     lo_flux_src_ndqp(i,k,l)%vf, &
                                                    lo_flux_gsrc_ndqp(i,k,l)%vf, &
                                                                     i, ix,iy,iz )
                            END DO
                        END DO
                    END DO
                ELSE
                    ! we do this (SHB)
                    ! Computing Riemann Solver Flux and Source Flux =================
                    DO k = ipsi%beg, ipsi%end, 2
                        DO j = ichi%beg, ichi%end, 2
                            CALL s_riemann_solver(     qR_prim_ndqp(i,j,k)%vf, &
                                                   dqR_prim_dx_ndqp(i,j,k)%vf, &
                                                   dqR_prim_dy_ndqp(i,j,k)%vf, &
                                                   dqR_prim_dz_ndqp(i,j,k)%vf, &
                                                     gm_alphaR_ndqp(i,j,k)%vf, &
                                                        kappaR_ndqp(i,j,k)%vf, &
                                                       qL_prim_ndqp(i,j,k)%vf, &
                                                   dqL_prim_dx_ndqp(i,j,k)%vf, &
                                                   dqL_prim_dy_ndqp(i,j,k)%vf, &
                                                   dqL_prim_dz_ndqp(i,j,k)%vf, &
                                                     gm_alphaL_ndqp(i,j,k)%vf, &
                                                        kappaL_ndqp(i,j,k)%vf, &
                                                          q_prim_qp(0,0,0)%vf, &
                                                          flux_ndqp(i,j,k)%vf, &
                                                      flux_src_ndqp(i,j,k)%vf, &
                                                     flux_gsrc_ndqp(i,j,k)%vf, &
                                                                  i, ix,iy,iz  )
                        END DO
                    END DO
                    print '(a)', 'SHB: after riemann solver flux and sources.'
                  
                    !print*, rhs_vf(1)%sf(:,0,0)
                    !print*, 'after riemann solver'

                    iv%beg = 1; iv%end = adv_idx%end
       
                    !won't do anything without flag
                    CALL s_average_cell_boundary_values(flux_ndqp(i,:,:))
                   
                    IF(ANY(Re_size > 0) .OR. We_size > 0) THEN
                        iv%beg = mom_idx%beg
                    ELSE
                        iv%beg = adv_idx%beg
                    END IF
                   
                    IF(riemann_solver /= 1) iv%end = adv_idx%beg
                   
                    !won't do anything without flag
                    CALL s_average_cell_boundary_values(flux_src_ndqp(i,:,:))
                    CALL s_average_cell_boundary_values(flux_gsrc_ndqp(i,:,:))
                   ! ===============================================================
                END IF


                IF (alt_soundspeed .OR. regularization) THEN
                    DO j = 0, m
                        DO k = 0, n
                            DO l = 0, p
                                blkmod1(j,k,l) = ((fluid_pp(1)%gamma+1d0)*q_prim_qp(0,0,0)%vf(E_idx)%sf(j,k,l) + &
                                                   fluid_pp(1)%pi_inf)/fluid_pp(1)%gamma
                                blkmod2(j,k,l) = ((fluid_pp(2)%gamma+1d0)*q_prim_qp(0,0,0)%vf(E_idx)%sf(j,k,l) + &
                                                   fluid_pp(2)%pi_inf)/fluid_pp(2)%gamma
                                alpha1(j,k,l) = q_cons_qp(0,0,0)%vf(adv_idx%beg)%sf(j,k,l)
                                
                                if (bubbles) then
                                    alpha2(j,k,l) = q_cons_qp(0,0,0)%vf(alf_idx-1)%sf(j,k,l)
                                else
                                    alpha2(j,k,l) = q_cons_qp(0,0,0)%vf(adv_idx%end)%sf(j,k,l)
                                end if
                                
                                Kterm(j,k,l) = alpha1(j,k,l)*alpha2(j,k,l)*(blkmod2(j,k,l)-blkmod1(j,k,l)) / &
                                                (alpha1(j,k,l)*blkmod2(j,k,l) + alpha2(j,k,l)*blkmod1(j,k,l))
                            END DO
                        END DO
                    END DO
                    if (bubbles) then
                        print*, 'max Kterm = ', maxval(abs(alpha1(:,:,:))), maxval(abs(alpha2(:,:,:))), &
                                            1.d0 - maxval(abs( q_cons_qp(0,0,0)%vf(alf_idx-2)%sf(:,:,:) + &
                                                               q_cons_qp(0,0,0)%vf(alf_idx-1)%sf(:,:,:) + &
                                                               q_cons_qp(0,0,0)%vf(alf_idx-0)%sf(:,:,:) ) ), &
                                            maxval(abs(Kterm(:,:,:)))
                    end if
                END IF



                ! adding the components to the RHS, starting with x-direction !
                ! RHS Contribution in x-direction ===============================
                IF(i == 1) THEN
                    !print '(a)', 'SHB: before x component rhs'
                  
                    ! Compute upwind slope and flux limiter function value if TVD
                    ! flux limiter is chosen
                    if (tvd_rhs_flux) call s_get_tvd_flux(q_cons_vf, q_prim_vf, rhs_vf,i)

                    ! Applying characteristic boundary conditions
                    IF(bc_x%beg <= -5) THEN
                        CALL s_cbc( q_prim_qp(0,0,0)%vf, flux_ndqp(i,0,0)%vf, &
                                    flux_src_ndqp(i,0,0)%vf, i, -1, ix,iy,iz  )
                    END IF
                    
                    IF(bc_x%end <= -5) THEN
                        CALL s_cbc( q_prim_qp(0,0,0)%vf, flux_ndqp(i,0,0)%vf, &
                                    flux_src_ndqp(i,0,0)%vf, i,  1, ix,iy,iz  )
                    END IF
            
                    ! We do this
                    ! Applying the Riemann fluxes
                    !print*, 'apply riemann fluxes'
                    DO j = 1, sys_size
                        DO k = 0, m
                            rhs_vf(j)%sf(k,:,:) = 1d0/dx(k) * &
                            ( flux_ndqp(i,0,0)%vf(j)%sf(k-1,0:n,0:p) &
                            - flux_ndqp(i,0,0)%vf(j)%sf( k ,0:n,0:p) )
                        END DO
                    END DO
    
                    !print*, 'apply riemann fluxes 2'
                    ! Applying source terms to the RHS of the advection equations
                    IF(riemann_solver == 1) THEN
                        DO j = adv_idx%beg, adv_idx%end
                            DO k = 0, m
                                rhs_vf(j)%sf(k,:,:) = &
                                rhs_vf(j)%sf(k,:,:) + 1d0/dx(k) * &
                                q_prim_qp(0,0,0)%vf(cont_idx%end+i)%sf(k,0:n,0:p) * &
                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(k-1,0:n,0:p) &
                                - flux_src_ndqp(i,0,0)%vf(j)%sf( k ,0:n,0:p) )
                            END DO
                        END DO
                    ELSE
                        DO j = adv_idx%beg, adv_idx%end
                            IF (alt_soundspeed .OR. regularization) THEN
                                IF ( adv_alphan .AND. (j == adv_idx%end) .and. (bubbles .neqv. .TRUE.)) THEN
                                    !adv_idx%end, -k div(u)
                                    DO k = 0, m
                                        rhs_vf(j)%sf(k,:,:) = &
                                        rhs_vf(j)%sf(k,:,:) + 1d0/dx(k) * &
                                        (q_cons_qp(0,0,0)%vf(j)%sf(k,0:n,0:p) - Kterm(k,:,:)) * &
                                        ( flux_src_ndqp(i,0,0)%vf(j)%sf( k ,0:n,0:p) &
                                        - flux_src_ndqp(i,0,0)%vf(j)%sf(k-1,0:n,0:p) )
                                    END DO
                                ELSE IF ( adv_alphan .AND. (j == adv_idx%beg) .and. (bubbles .neqv. .TRUE.)) THEN
                                    !adv_idx%beg, +k div(u)
                                    DO k = 0, m
                                        rhs_vf(j)%sf(k,:,:) = &
                                        rhs_vf(j)%sf(k,:,:) + 1d0/dx(k) * &
                                        (q_cons_qp(0,0,0)%vf(j)%sf(k,0:n,0:p) + Kterm(k,:,:)) * &
                                        ( flux_src_ndqp(i,0,0)%vf(j)%sf( k ,0:n,0:p) &
                                        - flux_src_ndqp(i,0,0)%vf(j)%sf(k-1,0:n,0:p) )
                                    END DO
                                ELSE IF ( adv_alphan .AND. (j == adv_idx%beg) .and. bubbles) THEN
                                    !liquid part, +k div(u)
                                    DO k = 0, m
                                        rhs_vf(j)%sf(k,:,:) = &
                                        rhs_vf(j)%sf(k,:,:) + 1d0/dx(k) * &
                                        (q_cons_qp(0,0,0)%vf(j)%sf(k,0:n,0:p) + Kterm(k,:,:)) * &
                                        ( flux_src_ndqp(i,0,0)%vf(j)%sf( k ,0:n,0:p) &
                                        - flux_src_ndqp(i,0,0)%vf(j)%sf(k-1,0:n,0:p) )
                                    END DO
                                ELSE IF ( adv_alphan .AND. (j == alf_idx-1) .and. bubbles) THEN
                                    !resolved gas, -k div(u)
                                    DO k = 0, m
                                        rhs_vf(j)%sf(k,:,:) = &
                                        rhs_vf(j)%sf(k,:,:) + 1d0/dx(k) * &
                                        (q_cons_qp(0,0,0)%vf(j)%sf(k,0:n,0:p) - Kterm(k,:,:)) * &
                                        ( flux_src_ndqp(i,0,0)%vf(j)%sf( k ,0:n,0:p) &
                                        - flux_src_ndqp(i,0,0)%vf(j)%sf(k-1,0:n,0:p) )
                                    END DO
                                ELSE IF ( adv_alphan .AND. (j == alf_idx) .and. bubbles) THEN
                                    !bubble part, no k div(u)
                                    DO k = 0, m
                                        rhs_vf(j)%sf(k,:,:) = &
                                        rhs_vf(j)%sf(k,:,:) + 1d0/dx(k) * &
                                        q_cons_qp(0,0,0)%vf(j)%sf(k,0:n,0:p) * &
                                        ( flux_src_ndqp(i,0,0)%vf(j)%sf( k ,0:n,0:p) &
                                        - flux_src_ndqp(i,0,0)%vf(j)%sf(k-1,0:n,0:p) )
                                    END DO
                              END IF
                            ELSE
                                !no k \div u, just adds other part of the transport equation
                                DO k = 0, m
                                    rhs_vf(j)%sf(k,:,:) = &
                                        rhs_vf(j)%sf(k,:,:) + 1d0/dx(k) * &
                                        q_cons_qp(0,0,0)%vf(j)%sf(k,0:n,0:p) * &
                                        ( flux_src_ndqp(i,0,0)%vf(j)%sf( k ,0:n,0:p) &
                                        - flux_src_ndqp(i,0,0)%vf(j)%sf(k-1,0:n,0:p) )
                                END DO
                            END IF
                        END DO
                    END IF

                    If (bubbles) then
                        call s_get_divergence(i,q_prim_vf,divu)
                        call s_compute_bubble_source(i,q_prim_vf,q_cons_vf,divu, &
                                bub_adv_src, bub_r_src, bub_v_src, bub_p_src, bub_m_src)
                        
                                              rhs_vf( alf_idx )%sf(:,:,:) = rhs_vf( alf_idx )%sf(:,:,:) + bub_adv_src(:,:,:)
                        if ( num_fluids >1 )  rhs_vf(adv_idx%beg)%sf(:,:,:) = rhs_vf(adv_idx%beg)%sf(:,:,:) - bub_adv_src(:,:,:)

                        do k = 1,nb
                            rhs_vf(bub_idx%rs(k))%sf(:,:,:) = rhs_vf(bub_idx%rs(k))%sf(:,:,:) + bub_r_src(k,:,:,:)
                            rhs_vf(bub_idx%vs(k))%sf(:,:,:) = rhs_vf(bub_idx%vs(k))%sf(:,:,:) + bub_v_src(k,:,:,:)
                            if (polytropic .neqv. .TRUE.) then
                                rhs_vf(bub_idx%ps(k))%sf(:,:,:) = rhs_vf(bub_idx%ps(k))%sf(:,:,:) + bub_p_src(k,:,:,:)
                                rhs_vf(bub_idx%ms(k))%sf(:,:,:) = rhs_vf(bub_idx%ms(k))%sf(:,:,:) + bub_m_src(k,:,:,:)
                            end if
                        end do
                   end if

                   !print*, 'before monopole calls'
                   If (monopole) then
                        mono_mass_src = 0d0; mono_mom_src = 0d0; mono_e_src = 0d0;
                        do j = 1,num_mono
                            call s_get_monopole(i,q_prim_vf,t_step,mono(j))
                        end do
                        do k = cont_idx%beg,cont_idx%end
                            rhs_vf(k)%sf(:,:,:) = rhs_vf(k)%sf(:,:,:) + mono_mass_src(:,:,:)
                        end do
                        do k = mom_idx%beg,mom_idx%end
                            rhs_vf(k)%sf(:,:,:) = rhs_vf(k)%sf(:,:,:) + mono_mom_src(k-cont_idx%end,:,:,:)
                        end do
                        rhs_vf(E_idx)%sf(:,:,:) = rhs_vf(E_idx)%sf(:,:,:) + mono_e_src(:,:,:)
                    end if

                    if (bubbles .and. num_fluids == 3 .and. bubble_switch) then
                        rhs_vf(alf_idx-1)%sf(:,:,:) = rhs_vf(alf_idx-1)%sf(:,:,:) - s_term(:,:,:)
                        rhs_vf( alf_idx )%sf(:,:,:) = rhs_vf( alf_idx )%sf(:,:,:) + s_term(:,:,:)
                    end if

                    ! Applying source terms to the RHS of the internal energy equations
                    IF(model_eqns == 3) THEN
                        DO j = 1, num_fluids
                            DO k = 0, m
                                rhs_vf(j+internalEnergies_idx%beg-1)%sf(k,:,:) = rhs_vf(j+internalEnergies_idx%beg-1)%sf(k,:,:) - 1d0/dx(k) * &
                                                        q_cons_qp(0,0,0)%vf(j+adv_idx%beg-1)%sf(k,0:n,0:p)*q_prim_qp(0,0,0)%vf(E_idx)%sf(k,0:n,0:p) * &
                                                        (flux_src_ndqp(i,0,0)%vf(adv_idx%beg)%sf( k ,0:n,0:p) - flux_src_ndqp(i,0,0)%vf(adv_idx%beg)%sf(k-1,0:n,0:p))
                            END DO
                        END DO
                    END IF

                    ! Applying the viscous and capillary source fluxes from the Riemann solver
                    IF(ANY(Re_size > 0) .OR. (We_size > 0 .AND. We_riemann_flux)) THEN
                        DO j = mom_idx%beg, E_idx
                            DO k = 0, m
                                rhs_vf(j)%sf(k,:,:) = &
                                rhs_vf(j)%sf(k,:,:) + 1d0/dx(k) * &
                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(k-1,0:n,0:p) &
                                - flux_src_ndqp(i,0,0)%vf(j)%sf( k ,0:n,0:p) )
                            END DO
                        END DO
                    END IF

                    ! Applying the capillary stress tensor calculated outside the Riemann solver
                    IF (We_size > 0 .AND. We_rhs_flux) THEN
                        IF (p > 0) THEN
                            CALL s_compute_capillary_stress_tensor(i, q_prim_qp(0,0,0)%vf, &
                                            dq_prim_dx_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                            dq_prim_dy_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                            dq_prim_dz_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                                                      gm_alpha_qp(0,0,0)%vf)
                        ELSE
                            CALL s_compute_capillary_stress_tensor(i, q_prim_qp(0,0,0)%vf, &
                                            dq_prim_dx_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                            dq_prim_dy_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                            dq_prim_dy_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                                                      gm_alpha_qp(0,0,0)%vf)
                        END IF
                        ! Divergence of capillary stress tensor computed using finite differences
                        DO j = mom_idx%beg, E_idx
                            DO k = 0, m
                                rhs_vf(j)%sf(k,:,:) = &
                                rhs_vf(j)%sf(k,:,:) - 1d0/(x_cc(k+1) - x_cc(k-1)) * &
                                ( tau_We_vf(j)%sf(k+1,0:n,0:p) &
                                - tau_We_vf(j)%sf(k-1,0:n,0:p) )
                            END DO
                        END DO
                    END IF

                    ! Applying capillary source terms in momentum and energy equations
                    IF (We_size > 0 .AND. We_src) THEN
                        CALL s_compute_capillary_source(i,q_prim_vf)
                        rhs_vf(cont_idx%end+i)%sf(:,:,:) = rhs_vf(cont_idx%end+i)%sf(:,:,:) + We_mtm_src(:,:,:)
                        rhs_vf(E_idx)%sf(:,:,:) = rhs_vf(E_idx)%sf(:,:,:) + We_nrg_src(:,:,:)
                    END IF
                    

                    
               ! ===============================================================
               
               ! RHS Contribution in y-direction ===============================
                ELSEIF(i == 2) THEN
                    !print '(a)', 'SHB: before y component rhs'
                  
                    ! Compute upwind slope and flux limiter function value if TVD
                    ! flux limiter is chosen
                    
                    if (tvd_rhs_flux) call s_get_tvd_flux(q_cons_vf, q_prim_vf, rhs_vf,i)

                    ! Applying characteristic boundary conditions
                    IF(bc_y%beg <= -5 .AND. bc_y%beg /= -13) THEN
                        CALL s_cbc( q_prim_qp(0,0,0)%vf, flux_ndqp(i,0,0)%vf, &
                                    flux_src_ndqp(i,0,0)%vf, i, -1, ix,iy,iz  )
                    END IF
                    
                    IF(bc_y%end <= -5) THEN
                        CALL s_cbc( q_prim_qp(0,0,0)%vf, flux_ndqp(i,0,0)%vf, &
                                    flux_src_ndqp(i,0,0)%vf, i,  1, ix,iy,iz  )
                    END IF

                    ! Applying the Riemann fluxes
                    DO j = 1, sys_size
                        DO k = 0, n
                            rhs_vf(j)%sf(:,k,:) = & 
                            rhs_vf(j)%sf(:,k,:) + 1d0/dy(k) * &
                            ( flux_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) &
                            - flux_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) )
                        END DO
                    END DO

                    ! Applying source terms to the RHS of the advection equations
                    IF(riemann_solver == 1) THEN
                        DO j = adv_idx%beg, adv_idx%end
                            DO k = 0, n
                                rhs_vf(j)%sf(:,k,:) = &
                                rhs_vf(j)%sf(:,k,:) + 1d0/dy(k) * &
                                q_prim_qp(0,0,0)%vf(cont_idx%end+i)%sf(0:m,k,0:p) * &
                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) &
                                - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) )
                            END DO
                        END DO
                    ELSE
                        DO j = adv_idx%beg, adv_idx%end
                            DO k = 0, n
                                IF (alt_soundspeed .OR. regularization) THEN
                                    IF ( adv_alphan .AND. (j == adv_idx%beg) .and. bubbles) THEN
                                        !liquid part, +k div(u)
                                        rhs_vf(j)%sf(:,k,:) = &
                                            rhs_vf(j)%sf(:,k,:) + 1d0/dy(k) * &
                                            (q_cons_qp(0,0,0)%vf(j)%sf(0:m,k,0:p) + Kterm(:,k,:)) * &
                                            ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) &
                                            - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) )
                                    ELSE IF ( adv_alphan .AND. (j == alf_idx-1) .and. bubbles) THEN
                                        !resolved gas, -k div(u)
                                        rhs_vf(j)%sf(:,k,:) = &
                                            rhs_vf(j)%sf(:,k,:) + 1d0/dy(k) * &
                                            (q_cons_qp(0,0,0)%vf(j)%sf(0:m,k,0:p) - Kterm(:,k,:)) * &
                                            ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) &
                                            - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) )
                                    ELSE IF ( adv_alphan .AND. (j == alf_idx) .and. bubbles) THEN
                                        !bubble part, no k div(u)
                                        rhs_vf(j)%sf(:,k,:) = &
                                            rhs_vf(j)%sf(:,k,:) + 1d0/dy(k) * &
                                            q_cons_qp(0,0,0)%vf(j)%sf(0:m,k,0:p) * &
                                            ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) &
                                            - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) )
                                    ELSE IF (adv_alphan .AND. (j == adv_idx%end)) THEN
                                        rhs_vf(j)%sf(:,k,:) = &
                                            rhs_vf(j)%sf(:,k,:) + 1d0/dy(k) * &
                                            (q_cons_qp(0,0,0)%vf(j)%sf(0:m,k,0:p) - Kterm(:,k,:)) * &
                                            ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) &
                                            - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) )
                                        IF (cyl_coord) THEN
                                            rhs_vf(j)%sf(:,k,:) = &
                                                rhs_vf(j)%sf(:,k,:) - Kterm(:,k,:)/2d0/y_cc(k) * &
                                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) &
                                                + flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) )
                                        END IF
                                    ELSE
                                        rhs_vf(j)%sf(:,k,:) = &
                                            rhs_vf(j)%sf(:,k,:) + 1d0/dy(k) * &
                                            (q_cons_qp(0,0,0)%vf(j)%sf(0:m,k,0:p) + Kterm(:,k,:)) * &
                                            ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) &
                                              - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) )
                                        IF (cyl_coord) THEN
                                            rhs_vf(j)%sf(:,k,:) = &
                                                rhs_vf(j)%sf(:,k,:) + a_coeff(0:m,k,0:p)*Kterm(:,k,:)/2d0/y_cc(k) * &
                                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) &
                                                + flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) )
                                        END IF
                                    END IF
                                ELSE
                                    rhs_vf(j)%sf(:,k,:) = &
                                        rhs_vf(j)%sf(:,k,:) + 1d0/dy(k) * &
                                        q_cons_qp(0,0,0)%vf(j)%sf(0:m,k,0:p) * &
                                        ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) &
                                        - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) )
                                END IF
                            END DO
                        END DO
                    END IF

                    If (bubbles) then
                        call s_get_divergence(i,q_prim_vf,divu)
                        call s_compute_bubble_source(i,q_prim_vf,q_cons_vf,divu, &
                                bub_adv_src, bub_r_src, bub_v_src, bub_p_src, bub_m_src)

                                              rhs_vf( alf_idx )%sf(:,:,:) = rhs_vf( alf_idx )%sf(:,:,:) + bub_adv_src(:,:,:)
                        if ( num_fluids > 1)  rhs_vf(adv_idx%beg)%sf(:,:,:) = rhs_vf(adv_idx%beg)%sf(:,:,:) - bub_adv_src(:,:,:)

                        do k = 1,nb
                            rhs_vf(bub_idx%rs(k))%sf(:,:,:) = rhs_vf(bub_idx%rs(k))%sf(:,:,:) + bub_r_src(k,:,:,:)
                            rhs_vf(bub_idx%vs(k))%sf(:,:,:) = rhs_vf(bub_idx%vs(k))%sf(:,:,:) + bub_v_src(k,:,:,:)
                            if (polytropic .neqv. .TRUE.) then
                                rhs_vf(bub_idx%ps(k))%sf(:,:,:) = rhs_vf(bub_idx%ps(k))%sf(:,:,:) + bub_p_src(k,:,:,:)
                                rhs_vf(bub_idx%ms(k))%sf(:,:,:) = rhs_vf(bub_idx%ms(k))%sf(:,:,:) + bub_m_src(k,:,:,:)
                            end if
                        end do
                    end if

                    If (monopole) then
                        mono_mass_src = 0d0; mono_mom_src = 0d0; mono_e_src = 0d0;
                        do j = 1,num_mono
                            call s_get_monopole(i,q_prim_vf,t_step,mono(j))
                        end do
                        do k = cont_idx%beg,cont_idx%end
                            rhs_vf(k)%sf(:,:,:) = rhs_vf(k)%sf(:,:,:) + mono_mass_src(:,:,:)
                        end do
                        do k = mom_idx%beg,mom_idx%end
                            rhs_vf(k)%sf(:,:,:) = rhs_vf(k)%sf(:,:,:) + mono_mom_src(k-cont_idx%end,:,:,:)
                        end do
                        rhs_vf(E_idx)%sf(:,:,:) = rhs_vf(E_idx)%sf(:,:,:) + mono_e_src(:,:,:)
                    end if

                    ! Applying source terms to the RHS of the internal energy equations
                    IF(model_eqns == 3) THEN
                        DO j = 1, num_fluids
                            DO k = 0, n
                                rhs_vf(j+internalEnergies_idx%beg-1)%sf(:,k,:) = rhs_vf(j+internalEnergies_idx%beg-1)%sf(:,k,:) - 1d0/dy(k) * &
                                                        q_cons_qp(0,0,0)%vf(j+adv_idx%beg-1)%sf(0:m,k,0:p)*q_prim_qp(0,0,0)%vf(E_idx)%sf(0:m,k,0:p) * &
                                                        (flux_src_ndqp(i,0,0)%vf(adv_idx%beg)%sf(0:m, k ,0:p) - flux_src_ndqp(i,0,0)%vf(adv_idx%beg)%sf(0:m,k-1,0:p))
                                                        ! (flux_src_ndqp(i,0,0)%vf(j+adv_idx%beg-1)%sf(0:m, k ,0:p) - flux_src_ndqp(i,0,0)%vf(j+adv_idx%beg-1)%sf(0:m,k-1,0:p)) 
                                                        ! //KS// see remark above
                            END DO
                        END DO
                        ! Applying the additional geometrical inviscid Riemann source fluxes for the internal energy equations
                        ! using the average of velocities at cell boundaries
                        IF(cyl_coord) THEN
                            DO j = 1, num_fluids
                                DO k = 0, n
                                    rhs_vf(j+internalEnergies_idx%beg-1)%sf(:,k,:) = rhs_vf(j+internalEnergies_idx%beg-1)%sf(:,k,:) - 5d-1/y_cc(k) * &
                                                            q_cons_qp(0,0,0)%vf(j+adv_idx%beg-1)%sf(0:m,k,0:p)*q_prim_qp(0,0,0)%vf(E_idx)%sf(0:m,k,0:p) * &
                                                            (flux_src_ndqp(i,0,0)%vf(adv_idx%beg)%sf(0:m, k ,0:p) + flux_src_ndqp(i,0,0)%vf(adv_idx%beg)%sf(0:m,k-1,0:p)) 
                                    ! //KS// see remark above
                                END DO
                            END DO
                        END IF
                    END IF

                    ! Applying the geometrical inviscid Riemann source fluxes calculated as average
                    ! of values at cell boundaries
                    IF(cyl_coord) THEN
                        DO j = 1, sys_size
                            DO k = 0, n
                                rhs_vf(j)%sf(:,k,:) = & 
                                rhs_vf(j)%sf(:,k,:) - 5d-1/y_cc(k) * &
                                ( flux_gsrc_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) &
                                + flux_gsrc_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) )
                            END DO
                        END DO
                    END IF

                    ! Applying the viscous and capillary source fluxes from the Riemann solver
                    IF(ANY(Re_size > 0) .OR. (We_size > 0 .AND. We_riemann_flux)) THEN
                        DO j = mom_idx%beg, E_idx
                            IF (cyl_coord .AND. ((bc_y%beg == -2) .OR. (bc_y%beg == -13))) THEN
                                IF (p > 0) THEN
                                    CALL s_compute_viscous_stress_tensor(q_prim_qp(0,0,0)%vf, &
                                            dq_prim_dx_qp(0,0,0)%vf(mom_idx%beg:mom_idx%end), &
                                            dq_prim_dy_qp(0,0,0)%vf(mom_idx%beg:mom_idx%end), &
                                            dq_prim_dz_qp(0,0,0)%vf(mom_idx%beg:mom_idx%end) )
                                ELSE 
                                    CALL s_compute_viscous_stress_tensor(q_prim_qp(0,0,0)%vf, &
                                            dq_prim_dx_qp(0,0,0)%vf(mom_idx%beg:mom_idx%end), &
                                            dq_prim_dy_qp(0,0,0)%vf(mom_idx%beg:mom_idx%end), &
                                            dq_prim_dy_qp(0,0,0)%vf(mom_idx%beg:mom_idx%end) )
                                END IF
                                DO k = 1, n
                                    rhs_vf(j)%sf(:,k,:) = &
                                    rhs_vf(j)%sf(:,k,:) + 1d0/dy(k) * &
                                    ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) &
                                    - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) )
                                END DO
                                rhs_vf(j)%sf(:,0,:) = &
                                rhs_vf(j)%sf(:,0,:) + 1d0/(y_cc(1) - y_cc(-1)) * &
                                ( tau_Re_vf(j)%sf(0:m,-1,0:p) &
                                - tau_Re_vf(j)%sf(0:m, 1,0:p) )
                            ELSE
                                DO k = 0, n
                                    rhs_vf(j)%sf(:,k,:) = &
                                    rhs_vf(j)%sf(:,k,:) + 1d0/dy(k) * &
                                    ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) &
                                    - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) )
                                END DO
                            END IF
                        END DO
                        ! Applying the geometrical viscous Riemann source fluxes calculated as average
                        ! of values at cell boundaries
                        IF (cyl_coord) THEN
                            DO j = mom_idx%beg, E_idx
                                IF ((bc_y%beg == -2) .OR. (bc_y%beg == -13)) THEN
                                    DO k = 1, n
                                        rhs_vf(j)%sf(:,k,:) = &
                                        rhs_vf(j)%sf(:,k,:) - 5d-1/y_cc(k) * &
                                        ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) &
                                        + flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) )
                                    END DO
                                    rhs_vf(j)%sf(:,0,:) = &
                                    rhs_vf(j)%sf(:,0,:) - 1d0/y_cc(0) * &
                                    tau_Re_vf(j)%sf(0:m,0,0:p)
                                ELSE
                                    DO k = 0, n
                                        rhs_vf(j)%sf(:,k,:) = &
                                        rhs_vf(j)%sf(:,k,:) - 5d-1/y_cc(k) * &
                                        ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,k-1,0:p) &
                                        + flux_src_ndqp(i,0,0)%vf(j)%sf(0:m, k ,0:p) )
                                    END DO
                                END IF
                            END DO
                        END IF
                    END IF

                    ! Applying the capillary stress tensor calculated outside the Riemann solver
                    IF (We_size > 0 .AND. We_rhs_flux) THEN
                        IF (p > 0) THEN
                            CALL s_compute_capillary_stress_tensor(i, q_prim_qp(0,0,0)%vf, &
                                            dq_prim_dx_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                            dq_prim_dy_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                            dq_prim_dz_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                                                      gm_alpha_qp(0,0,0)%vf)
                        ELSE 
                            CALL s_compute_capillary_stress_tensor(i, q_prim_qp(0,0,0)%vf, &
                                            dq_prim_dx_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                            dq_prim_dy_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                            dq_prim_dy_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                                                      gm_alpha_qp(0,0,0)%vf)
                        END IF
                        ! Divergence of capillary stress tensor computed using finite differences
                        DO j = mom_idx%beg, E_idx
                            DO k = 0, n
                                rhs_vf(j)%sf(:,k,:) = &
                                rhs_vf(j)%sf(:,k,:) - 1d0/(y_cc(k+1) - y_cc(k-1)) * &
                                ( tau_We_vf(j)%sf(0:m,k+1,0:p) &
                                - tau_We_vf(j)%sf(0:m,k-1,0:p) )
                            END DO
                        END DO
                        ! Applying the capillary contribution to geometric source term
                        IF (cyl_coord) THEN
                            DO j = mom_idx%beg, E_idx
                                DO k = 0, n
                                    rhs_vf(j)%sf(:,k,:) = &
                                    rhs_vf(j)%sf(:,k,:) - tau_We_vf(j)%sf(0:m,k,0:p)/y_cc(k)
                                END DO
                            END DO
                        END IF
                    END IF
                    
                    ! Applying capillary source terms in momentum and energy equations
                    IF (We_size > 0 .AND. We_src) THEN
                        CALL s_compute_capillary_source(i,q_prim_vf)
                        rhs_vf(cont_idx%end+i)%sf(:,:,:) = rhs_vf(cont_idx%end+i)%sf(:,:,:) + We_mtm_src(:,:,:)
                        rhs_vf(E_idx)%sf(:,:,:) = rhs_vf(E_idx)%sf(:,:,:) + We_nrg_src(:,:,:)
                    END IF

                    ! Applying interface sharpening regularization source terms
                    IF (regularization .AND. num_dims == 2) THEN
                        CALL s_compute_regularization_source(i,q_prim_vf)
                        DO j = cont_idx%beg, adv_idx%end
                            rhs_vf(j)%sf(:,:,:) = rhs_vf(j)%sf(:,:,:) + reg_src_vf(j)%sf(:,:,:)
                        END DO
                    END IF


               ! ===============================================================
               
               ! RHS Contribution in z-direction ===============================
                ELSE
                    !print '(a)', 'SHB: before z component rhs'

                    ! Compute upwind slope and flux limiter function value if TVD
                    ! flux limiter is chosen
                    if (tvd_rhs_flux) call s_get_tvd_flux(q_cons_vf, q_prim_vf, rhs_vf,i)

                    ! Applying characteristic boundary conditions
                    IF(bc_z%beg <= -5) THEN
                        CALL s_cbc( q_prim_qp(0,0,0)%vf, flux_ndqp(i,0,0)%vf, &
                                    flux_src_ndqp(i,0,0)%vf, i, -1, ix,iy,iz  )
                    END IF
                    
                    IF(bc_z%end <= -5) THEN
                        CALL s_cbc( q_prim_qp(0,0,0)%vf, flux_ndqp(i,0,0)%vf, &
                                    flux_src_ndqp(i,0,0)%vf, i,  1, ix,iy,iz  )
                    END IF
                    
                    ! Applying the Riemann fluxes
                    DO j = 1, sys_size
                        IF (grid_geometry == 3) THEN
                            DO l = 0, n
                                DO k = 0, p
                                    rhs_vf(j)%sf(:,l,k) = & 
                                    rhs_vf(j)%sf(:,l,k) + 1d0/dz(k)/y_cc(l) * &
                                    ( flux_ndqp(i,0,0)%vf(j)%sf(0:m,l,k-1) &
                                    - flux_ndqp(i,0,0)%vf(j)%sf(0:m,l, k ) )
                                END DO
                            END DO
                        ELSE
                            DO k = 0, p
                                rhs_vf(j)%sf(:,:,k) = & 
                                rhs_vf(j)%sf(:,:,k) + 1d0/dz(k) * &
                                ( flux_ndqp(i,0,0)%vf(j)%sf(0:m,0:n,k-1) &
                                - flux_ndqp(i,0,0)%vf(j)%sf(0:m,0:n, k ) )
                            END DO
                        END IF
                    END DO

                    ! Applying source terms to the RHS of the advection equations
                    IF(riemann_solver == 1) THEN
                        DO j = adv_idx%beg, adv_idx%end
                            IF (grid_geometry == 3) THEN
                                DO l = 0, n
                                    DO k = 0, p
                                        rhs_vf(j)%sf(:,l,k) = &
                                        rhs_vf(j)%sf(:,l,k) + 1d0/dz(k)/y_cc(l) * &
                                        q_prim_qp(0,0,0)%vf(cont_idx%end+i)%sf(0:m,l,k) * &
                                        ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,l,k-1) &
                                        - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,l, k ) )
                                    END DO
                                END DO
                            ELSE
                                DO k = 0, p
                                    rhs_vf(j)%sf(:,:,k) = &
                                    rhs_vf(j)%sf(:,:,k) + 1d0/dz(k) * &
                                    q_prim_qp(0,0,0)%vf(cont_idx%end+i)%sf(0:m,0:n,k) * &
                                    ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n,k-1) &
                                    - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n, k ) )
                                END DO
                            END IF
                        END DO
                    ELSE
                        DO j = adv_idx%beg, adv_idx%end
                            IF (grid_geometry == 3) THEN
                                DO l = 0, n
                                    DO k = 0, p
                                        IF (alt_soundspeed .OR. regularization) THEN
                                            IF (adv_alphan .AND. j == adv_idx%end) THEN
                                                rhs_vf(j)%sf(:,l,k) = &
                                                rhs_vf(j)%sf(:,l,k) + 1d0/dz(k)/y_cc(l) * &
                                                (q_cons_qp(0,0,0)%vf(j)%sf(0:m,l,k) - a_coeff(0:m,l,k)*Kterm(:,l,k)) * &
                                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,l, k ) &
                                                - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,l,k-1) )
                                            ELSE
                                                rhs_vf(j)%sf(:,l,k) = &
                                                rhs_vf(j)%sf(:,l,k) + 1d0/dz(k)/y_cc(l) * &
                                                (q_cons_qp(0,0,0)%vf(j)%sf(0:m,l,k) + a_coeff(0:m,l,k)*Kterm(:,l,k)) * &
                                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,l, k ) &
                                                - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,l,k-1) )
                                            END IF
                                        ELSE
                                            rhs_vf(j)%sf(:,l,k) = &
                                            rhs_vf(j)%sf(:,l,k) + 1d0/dz(k)/y_cc(l) * &
                                            q_cons_qp(0,0,0)%vf(j)%sf(0:m,l,k) * &
                                            ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,l, k ) &
                                            - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,l,k-1) )
                                        END IF
                                    END DO
                                END DO
                            ELSE
                                DO k = 0, p
                                    IF (alt_soundspeed .OR. regularization) THEN
                                        IF ( adv_alphan .AND. (j == adv_idx%beg) .and. bubbles) THEN
                                            !liquid part, +k div(u)
                                            rhs_vf(j)%sf(:,:,k) = &
                                               rhs_vf(j)%sf(:,:,k) + 1d0/dz(k) * &
                                               (q_cons_qp(0,0,0)%vf(j)%sf(0:m,0:n,k) + Kterm(:,:,k)) * &
                                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n, k ) &
                                                - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n,k-1) )

                                        ELSE IF ( adv_alphan .AND. (j == alf_idx-1) .and. bubbles) THEN
                                            !resolved gas, -k div(u)
                                            rhs_vf(j)%sf(:,:,k) = &
                                                rhs_vf(j)%sf(:,:,k) + 1d0/dz(k) * &
                                                (q_cons_qp(0,0,0)%vf(j)%sf(0:m,0:n,k) - Kterm(:,:,k)) * &
                                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n, k ) &
                                                - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n,k-1) )
                                        ELSE IF ( adv_alphan .AND. (j == alf_idx) .and. bubbles) THEN
                                            !bubble part, no k div(u)
                                            rhs_vf(j)%sf(:,:,k) = &
                                                rhs_vf(j)%sf(:,:,k) + 1d0/dz(k) * &
                                                q_cons_qp(0,0,0)%vf(j)%sf(0:m,0:n,k) * &
                                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n, k ) &
                                                - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n,k-1) )                                            
                                        ELSE IF (adv_alphan .AND. j == adv_idx%end) THEN
                                            rhs_vf(j)%sf(:,:,k) = &
                                               rhs_vf(j)%sf(:,:,k) + 1d0/dz(k) * &
                                               (q_cons_qp(0,0,0)%vf(j)%sf(0:m,0:n,k) - Kterm(:,:,k)) * &
                                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n, k ) &
                                                - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n,k-1) )
                                        ELSE
                                            rhs_vf(j)%sf(:,:,k) = &
                                                rhs_vf(j)%sf(:,:,k) + 1d0/dz(k) * &
                                                (q_cons_qp(0,0,0)%vf(j)%sf(0:m,0:n,k) + Kterm(:,:,k)) * &
                                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n, k ) &
                                                - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n,k-1) )
                                        END IF
                                    ELSE
                                        rhs_vf(j)%sf(:,:,k) = &
                                            rhs_vf(j)%sf(:,:,k) + 1d0/dz(k) * &
                                            q_cons_qp(0,0,0)%vf(j)%sf(0:m,0:n,k) * &
                                            ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n, k ) &
                                            - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n,k-1) ) 
                                    END IF
                                END DO
                            END IF
                        END DO
                    END IF
                    
                    If (bubbles) then
                        call s_get_divergence(i,q_prim_vf,divu)
                        call s_compute_bubble_source(i,q_prim_vf,q_cons_vf,divu, &
                                bub_adv_src, bub_r_src, bub_v_src, bub_p_src, bub_m_src)

                                              rhs_vf( alf_idx )%sf(:,:,:) = rhs_vf( alf_idx )%sf(:,:,:) + bub_adv_src(:,:,:)
                        if ( num_fluids > 1)  rhs_vf(adv_idx%beg)%sf(:,:,:) = rhs_vf(adv_idx%beg)%sf(:,:,:) - bub_adv_src(:,:,:)

                        do k = 1,nb
                            rhs_vf(bub_idx%rs(k))%sf(:,:,:) = rhs_vf(bub_idx%rs(k))%sf(:,:,:) + bub_r_src(k,:,:,:)
                            rhs_vf(bub_idx%vs(k))%sf(:,:,:) = rhs_vf(bub_idx%vs(k))%sf(:,:,:) + bub_v_src(k,:,:,:)
                            if (polytropic .neqv. .TRUE.) then
                                rhs_vf(bub_idx%ps(k))%sf(:,:,:) = rhs_vf(bub_idx%ps(k))%sf(:,:,:) + bub_p_src(k,:,:,:)
                                rhs_vf(bub_idx%ms(k))%sf(:,:,:) = rhs_vf(bub_idx%ms(k))%sf(:,:,:) + bub_m_src(k,:,:,:)
                            end if
                        end do
                    end if 

                    If (monopole) then
                        mono_mass_src = 0d0; mono_mom_src = 0d0; mono_e_src = 0d0;
                        do j = 1,num_mono
                            call s_get_monopole(i,q_prim_vf,t_step,mono(j))
                        end do
                        do k = cont_idx%beg,cont_idx%end
                            rhs_vf(k)%sf(:,:,:) = rhs_vf(k)%sf(:,:,:) + mono_mass_src(:,:,:)
                        end do
                        do k = mom_idx%beg,mom_idx%end
                            rhs_vf(k)%sf(:,:,:) = rhs_vf(k)%sf(:,:,:) + mono_mom_src(k-cont_idx%end,:,:,:)
                        end do
                        rhs_vf(E_idx)%sf(:,:,:) = rhs_vf(E_idx)%sf(:,:,:) + mono_e_src(:,:,:)
                    end if

                    ! Applying source terms to the RHS of the internal energy equations
                    IF(model_eqns == 3) THEN
                        DO j = 1, num_fluids
                            DO k = 0, p
                                rhs_vf(j+internalEnergies_idx%beg-1)%sf(:,:,k) = rhs_vf(j+internalEnergies_idx%beg-1)%sf(:,:,k) - 1d0/dz(k) * &
                                                        q_cons_qp(0,0,0)%vf(j+adv_idx%beg-1)%sf(0:m,0:n,k)*q_prim_qp(0,0,0)%vf(E_idx)%sf(0:m,0:n,k) * &
                                                        (flux_src_ndqp(i,0,0)%vf(adv_idx%beg)%sf(0:m,0:n, k ) - flux_src_ndqp(i,0,0)%vf(adv_idx%beg)%sf(0:m,0:n,k-1))
                                                        ! (flux_src_ndqp(i,0,0)%vf(j+adv_idx%beg-1)%sf(0:m,0:n, k ) - flux_src_ndqp(i,0,0)%vf(j+adv_idx%beg-1)%sf(0:m,0:n,k-1)) 
                                                        ! //KS// see remark above
                            END DO
                        END DO
                    END IF

                    ! Applying the geometrical inviscid Riemann source fluxes calculated as average
                    ! of values at cell boundaries
                    IF(grid_geometry == 3) THEN
                        DO j = 1, sys_size
                            DO l = 0, n
                                DO k = 0, p
                                    rhs_vf(j)%sf(:,l,k) = & 
                                    rhs_vf(j)%sf(:,l,k) - 5d-1/y_cc(l) * &
                                    ( flux_gsrc_ndqp(i,0,0)%vf(j)%sf(0:m,l,k-1) &
                                    + flux_gsrc_ndqp(i,0,0)%vf(j)%sf(0:m,l, k ) )
                                END DO
                            END DO
                        END DO
                    END IF

                    ! Applying the viscous and capillary source fluxes from the Riemann solver
                    IF(ANY(Re_size > 0) .OR. (We_size > 0 .AND. We_riemann_flux)) THEN
                        DO j = mom_idx%beg, E_idx
                            DO k = 0, p
                                rhs_vf(j)%sf(:,:,k) = &
                                rhs_vf(j)%sf(:,:,k) + 1d0/dz(k) * &
                                ( flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n,k-1) &
                                - flux_src_ndqp(i,0,0)%vf(j)%sf(0:m,0:n, k ) )
                            END DO
                        END DO
                        ! Modifying momentum components of geometric source term
                        IF (grid_geometry == 3) THEN
                            DO k = 0, p
                                rhs_vf(mom_idx%beg+1)%sf(:,:,k) = &
                                rhs_vf(mom_idx%beg+1)%sf(:,:,k) + 5d-1 * &
                                ( flux_src_ndqp(i,0,0)%vf(mom_idx%end)%sf(0:m,0:n,k-1) &
                                + flux_src_ndqp(i,0,0)%vf(mom_idx%end)%sf(0:m,0:n, k ) )
                        
                                rhs_vf(mom_idx%end)%sf(:,:,k) = &
                                rhs_vf(mom_idx%end)%sf(:,:,k) - 5d-1 * &
                                ( flux_src_ndqp(i,0,0)%vf(mom_idx%beg+1)%sf(0:m,0:n,k-1) &
                                + flux_src_ndqp(i,0,0)%vf(mom_idx%beg+1)%sf(0:m,0:n, k ) )
                            END DO
                        END IF
                    END IF
                    
                    ! Applying the capillary stress tensor calculated outside the Riemann solver
                    IF (We_size > 0 .AND. We_rhs_flux) THEN
                        IF (p > 0) THEN
                            CALL s_compute_capillary_stress_tensor(i, q_prim_qp(0,0,0)%vf, &
                                            dq_prim_dx_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                            dq_prim_dy_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                            dq_prim_dz_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                                                      gm_alpha_qp(0,0,0)%vf)
                        ELSE 
                            CALL s_compute_capillary_stress_tensor(i, q_prim_qp(0,0,0)%vf, &
                                            dq_prim_dx_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                            dq_prim_dy_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                            dq_prim_dy_qp(0,0,0)%vf(adv_idx%beg:adv_idx%end), &
                                                                      gm_alpha_qp(0,0,0)%vf)
                        END IF
                        ! Divergence of capillary stress tensor computed using finite differences
                        DO j = mom_idx%beg, E_idx
                            IF (grid_geometry == 3) THEN
                                DO l = 0, n
                                    DO k = 0, p
                                        rhs_vf(j)%sf(:,l,k) = &
                                        rhs_vf(j)%sf(:,l,k) - 1d0/(z_cc(k+1) - z_cc(k-1))/y_cc(l) * &
                                        ( tau_We_vf(j)%sf(0:m,l,k+1) &
                                        - tau_We_vf(j)%sf(0:m,l,k-1) )
                                    END DO
                                END DO
                            ELSE
                                DO k = 0, p
                                    rhs_vf(j)%sf(:,:,k) = &
                                    rhs_vf(j)%sf(:,:,k) - 1d0/(z_cc(k+1) - z_cc(k-1)) * &
                                    ( tau_We_vf(j)%sf(0:m,0:n,k+1) &
                                    - tau_We_vf(j)%sf(0:m,0:n,k-1) )
                                END DO
                            END IF
                        END DO
                    END IF
                    
                    ! Applying capillary source terms in momentum and energy equations
                    IF (We_size > 0 .AND. We_src) THEN
                        CALL s_compute_capillary_source(i,q_prim_vf)
                        rhs_vf(cont_idx%end+i)%sf(:,:,:) = rhs_vf(cont_idx%end+i)%sf(:,:,:) + We_mtm_src(:,:,:)
                        rhs_vf(E_idx)%sf(:,:,:) = rhs_vf(E_idx)%sf(:,:,:) + We_nrg_src(:,:,:)
                    END IF
                  
                    ! Applying interface sharpening regularization source terms
                    IF (regularization .AND. num_dims == 3) THEN
                        CALL s_compute_regularization_source(i,q_prim_vf)
                        DO j = cont_idx%beg, adv_idx%end
                            rhs_vf(j)%sf(:,:,:) = rhs_vf(j)%sf(:,:,:) + reg_src_vf(j)%sf(:,:,:)
                        END DO
                    END IF

                END IF
               ! ===============================================================
               
            END DO
            ! END: Dimensional Splitting Loop ==================================
           
            
            ! Disassociation of Working Variables ==============================
            DO i = 1, sys_size
                NULLIFY(q_cons_qp(0,0,0)%vf(i)%sf, q_prim_qp(0,0,0)%vf(i)%sf)
            END DO
            ! ==================================================================
            
            
        END SUBROUTINE s_compute_rhs ! -----------------------------------------
        
        
        
        
        
        SUBROUTINE s_compute_flux_lim(top,bottom,flux_lim_func) ! ---------------
        ! Description: This subroutine takes the consecutive changes in volume fraction
        !   at a cell boundary and computes the desired slope and flux limiter
        !   function value at the cell boundary

            REAL(KIND(0d0)), INTENT(INOUT) :: top, bottom
            REAL(KIND(0d0)) :: slope
            REAL(KIND(0d0)), INTENT(OUT) :: flux_lim_func

            ! Limit the flux limiter to be applied only where the change in
            ! volume fraction is greater than machine precision so that insignificant
            ! fluctuations do not trip the limiter
            IF (ABS(top) < 1d-8) top = 0d0
            IF (ABS(bottom) < 1d-8) bottom = 0d0
            ! If top = bottom, then cell is in a smooth region of the flow
            ! and the high order flux should be used. Also ensures that areas
            ! of no change in volume fraction (0/0) use the high order flux
            IF (top == bottom) THEN
                slope = 1d0
            ELSE
                slope = (top*bottom)/MAX(bottom**2d0,sgm_eps)
            END IF

            ! Flux limiter function
            IF (flux_lim == 1) THEN ! MINMOD (MM)
                flux_lim_func = MAX(0d0,MIN(1d0,slope))
            ELSEIF (flux_lim == 2) THEN ! MUSCL (MC)
                flux_lim_func = MAX(0d0,MIN(2d0*slope,5d-1*(1d0+slope),2d0))
            ELSEIF (flux_lim == 3) THEN ! OSPRE (OP)
                flux_lim_func = (15d-1*(slope**2d0+slope))/(slope**2d0+slope+1d0)
            ELSEIF (flux_lim == 4) THEN ! SUPERBEE (SB)
                flux_lim_func = MAX(0d0,MIN(1d0,2d0*slope),MIN(slope,2d0))
            ELSEIF (flux_lim == 5) THEN ! SWEBY (SW) (beta = 1.5)
                flux_lim_func = MAX(0d0,MIN(15d-1*slope,1d0),MIN(slope,15d-1))
            ELSEIF (flux_lim == 6) THEN ! VAN ALBADA (VA)
                flux_lim_func = (slope**2d0+slope)/(slope**2d0+1d0)
            ELSEIF (flux_lim == 7) THEN ! VAN LEER (VL)
                flux_lim_func = (ABS(slope) + slope)/(1d0 + ABS(slope))
            END IF
        END SUBROUTINE s_compute_flux_lim ! ------------------------------------





        SUBROUTINE s_compute_viscous_stress_tensor(q_prim_vf,grad_x_vf,grad_y_vf,grad_z_vf) ! ---
        ! Description: The purpose of this subroutine is to compute the viscous 
        !              stress tensor for the cells directly next to the axis in
        !              cylindrical coordinates. This is necessary to avoid the
        !              1/r singularity that arises at the cell boundary coinciding
        !              with the axis, i.e., y_cb(-1) = 0.

            ! Cell-average primitive variables and derivatives
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: q_prim_vf
            TYPE(scalar_field), DIMENSION(num_dims), INTENT(IN) :: grad_x_vf, grad_y_vf, grad_z_vf

            ! Mixture variables
            REAL(KIND(0d0)) :: rho_visc, gamma_visc, pi_inf_visc
            REAL(KIND(0d0)), DIMENSION(2) :: Re_visc
            REAL(KIND(0d0)), DIMENSION(1:num_fluids,1:num_fluids) :: We_visc

            ! Capillary stress tensor components
            REAL(KIND(0d0)), DIMENSION(num_dims,num_dims) :: tau_Re

            TYPE(bounds_info) :: ix,iy,iz

            ! Generic loop iterator
            INTEGER :: i,j,k,l

            ix%beg = -buff_size; iy%beg = 0; iz%beg = 0
            IF(n > 0) iy%beg = -buff_size; IF(p > 0) iz%beg = -buff_size
            ix%end = m - ix%beg; iy%end = n - iy%beg; iz%end = p - iz%beg

            DO i = mom_idx%beg, E_idx
                tau_Re_vf(i)%sf = 0d0
            END DO

            IF (Re_size(1) > 0) THEN    ! Shear stresses
                DO l = iz%beg, iz%end
                    DO k = -1, 1
                        DO j = ix%beg, ix%end

                            CALL s_convert_to_mixture_variables(    q_prim_vf, rho_visc, &
                                                                gamma_visc, pi_inf_visc, &
                                                                Re_visc, We_visc, j,k,l  )

                            tau_Re(2,1) = (grad_y_vf(1)%sf(j,k,l) + &
                                           grad_x_vf(2)%sf(j,k,l))/ &
                                           Re_visc(1)

                            tau_Re(2,2) = ( 4d0*grad_y_vf(2)%sf(j,k,l) &
                                          - 2d0*grad_x_vf(1)%sf(j,k,l) &
                                          - 2d0*q_prim_vf(mom_idx%beg+1)%sf(j,k,l)/y_cc(k) ) / &
                                          ( 3d0*Re_visc(1) )

                            DO i = 1,2
                                tau_Re_vf(cont_idx%end+i)%sf(j,k,l) = &
                                tau_Re_vf(cont_idx%end+i)%sf(j,k,l) - &
                                tau_Re(2,i)

                                tau_Re_vf(E_idx)%sf(j,k,l) = &
                                tau_Re_vf(E_idx)%sf(j,k,l) - &
                                q_prim_vf(cont_idx%end+i)%sf(j,k,l) * tau_Re(2,i)
                            END DO

                        END DO
                    END DO
                END DO
            END IF

            IF (Re_size(2) > 0) THEN    ! Bulk stresses
                DO l = iz%beg, iz%end
                    DO k = -1, 1
                        DO j = ix%beg, ix%end

                            CALL s_convert_to_mixture_variables(    q_prim_vf, rho_visc, &
                                                                gamma_visc, pi_inf_visc, &
                                                                Re_visc, We_visc, j,k,l  )

                            tau_Re(2,2) = (grad_x_vf(1)%sf(j,k,l) + &
                                           grad_y_vf(2)%sf(j,k,l) + &
                                           q_prim_vf(mom_idx%beg+1)%sf(j,k,l)/y_cc(k) ) / &
                                           Re_visc(2)

                            tau_Re_vf(mom_idx%beg+1)%sf(j,k,l) = &
                            tau_Re_vf(mom_idx%beg+1)%sf(j,k,l) - &
                            tau_Re(2,2)

                            tau_Re_vf(E_idx)%sf(j,k,l) = &
                            tau_Re_vf(E_idx)%sf(j,k,l) - &
                            q_prim_vf(mom_idx%beg+1)%sf(j,k,l) * tau_Re(2,2)

                        END DO
                    END DO
                END DO
            END IF

            IF (p == 0) RETURN

            IF (Re_size(1) > 0) THEN    ! Shear stresses
                DO l = iz%beg, iz%end
                    DO k = -1, 1
                        DO j = ix%beg, ix%end

                            CALL s_convert_to_mixture_variables(    q_prim_vf, rho_visc, &
                                                                gamma_visc, pi_inf_visc, &
                                                                Re_visc, We_visc, j,k,l  )

                            tau_Re(2,2) = -(2d0/3d0)*grad_z_vf(3)%sf(j,k,l) / y_cc(k) / &
                                          Re_visc(1)

                            tau_Re(2,3) = ((grad_z_vf(2)%sf(j,k,l) - &
                                          q_prim_vf(mom_idx%end)%sf(j,k,l)) / &
                                          y_cc(k) + grad_y_vf(3)%sf(j,k,l) ) / &
                                          Re_visc(1)

                            DO i = 2,3
                                tau_Re_vf(cont_idx%end+i)%sf(j,k,l) = &
                                tau_Re_vf(cont_idx%end+i)%sf(j,k,l) - &
                                tau_Re(2,i)

                                tau_Re_vf(E_idx)%sf(j,k,l) = &
                                tau_Re_vf(E_idx)%sf(j,k,l) - &
                                q_prim_vf(cont_idx%end+i)%sf(j,k,l) * tau_Re(2,i)
                            END DO

                        END DO
                    END DO
                END DO
            END IF

            IF (Re_size(2) > 0) THEN    ! Bulk stresses
                DO l = iz%beg, iz%end
                    DO k = -1, 1
                        DO j = ix%beg, ix%end

                            tau_Re(2,2) = grad_z_vf(3)%sf(j,k,l) / y_cc(k) / &
                                          Re_visc(2)

                            tau_Re_vf(mom_idx%beg+1)%sf(j,k,l) = &
                            tau_Re_vf(mom_idx%beg+1)%sf(j,k,l) - &
                            tau_Re(2,2)

                            tau_Re_vf(E_idx)%sf(j,k,l) = &
                            tau_Re_vf(E_idx)%sf(j,k,l) - &
                            q_prim_vf(mom_idx%beg+1)%sf(j,k,l) * tau_Re(2,2)

                        END DO
                    END DO
                END DO
            END IF

        END SUBROUTINE s_compute_viscous_stress_tensor ! ----------------------------------------





        SUBROUTINE s_compute_capillary_stress_tensor(norm_dir,q_prim_vf,grad_x_vf,grad_y_vf,grad_z_vf,norm_vf) ! -------
        ! Description: The purpose of this procedure is to calculate the capillary
        !              stress tensor at the cell centers in the non-conservative 
        !              formulation of the equations.

            ! Dimensional split index
            INTEGER, INTENT(IN) :: norm_dir

            ! Cell-average primitive variables and derivatives
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: q_prim_vf
            TYPE(scalar_field), DIMENSION(num_fluids), INTENT(IN) :: norm_vf, grad_x_vf, grad_y_vf, grad_z_vf

            ! Mixture variables
            REAL(KIND(0d0)) :: rho_cap, gamma_cap, pi_inf_cap
            REAL(KIND(0d0)), DIMENSION(2) :: Re_cap
            REAL(KIND(0d0)), DIMENSION(1:num_fluids,1:num_fluids) :: We_cap

            ! Capillary stress tensor components
            REAL(KIND(0d0)), DIMENSION(num_dims,num_dims) :: tau_We

            TYPE(bounds_info) :: ix,iy,iz

            ! Generic loop iterator
            INTEGER :: i,j,k,l,r

            ix%beg = -buff_size; iy%beg = 0; iz%beg = 0
            IF(n > 0) iy%beg = -buff_size; IF(p > 0) iz%beg = -buff_size
            ix%end = m - ix%beg; iy%end = n - iy%beg; iz%end = p - iz%beg

            DO i = mom_idx%beg, E_idx
                tau_We_vf(i)%sf = 0d0
            END DO

            ! Capillary stresses in x-direction =============================
            IF (norm_dir == 1) THEN
                DO i = 1, We_size
                    DO l = iz%beg, iz%end
                        DO k = iy%beg, iy%end
                            DO j = ix%beg, ix%end

                                CALL s_convert_to_mixture_variables(   q_prim_vf, rho_cap, &
                                                                    gamma_cap, pi_inf_cap, &
                                                                    Re_cap, We_cap, j,k,l  )

                                tau_We(1,1) =  1d0/ We_cap(We_idx(i,1),We_idx(i,2)) * &
                                                   ( norm_vf(We_idx(i,2))%sf(j,k,l) - &
                                                   grad_x_vf(We_idx(i,2))%sf(j,k,l) * &
                                                   grad_x_vf(We_idx(i,2))%sf(j,k,l) / &
                                                 MAX(norm_vf(We_idx(i,2))%sf(j,k,l),sgm_eps))

                                tau_We(1,2) = -1d0/ We_cap(We_idx(i,1),We_idx(i,2)) * &
                                                   grad_x_vf(We_idx(i,2))%sf(j,k,l) * &
                                                   grad_y_vf(We_idx(i,2))%sf(j,k,l) / &
                                                 MAX(norm_vf(We_idx(i,2))%sf(j,k,l),sgm_eps)

                                DO r = 1,2
                                    tau_We_vf(cont_idx%end+r)%sf(j,k,l) = &
                                    tau_We_vf(cont_idx%end+r)%sf(j,k,l) - &
                                    tau_We(1,r)
                                END DO

                                tau_We(1,1) = -1d0/ We_cap(We_idx(i,1),We_idx(i,2)) * &
                                                   grad_x_vf(We_idx(i,2))%sf(j,k,l) * &
                                                   grad_x_vf(We_idx(i,2))%sf(j,k,l) / &
                                                 MAX(norm_vf(We_idx(i,2))%sf(j,k,l),sgm_eps)

                                DO r = 1,2
                                    tau_We_vf(E_idx)%sf(j,k,l) = &
                                    tau_We_vf(E_idx)%sf(j,k,l) - &
                                    q_prim_vf(cont_idx%end+r)%sf(j,k,l) * tau_We(1,r)
                                END DO

                            END DO
                        END DO
                    END DO
                END DO

                IF (p > 0) THEN
                    DO i = 1, We_size
                        DO l = iz%beg, iz%end
                            DO k = iy%beg, iy%end
                                DO j = ix%beg, ix%end
    
                                    CALL s_convert_to_mixture_variables(   q_prim_vf, rho_cap, &
                                                                        gamma_cap, pi_inf_cap, &
                                                                        Re_cap, We_cap, j,k,l  )
    
                                    tau_We(1,3) = -1d0/ We_cap(We_idx(i,1),We_idx(i,2)) * &
                                                       grad_x_vf(We_idx(i,2))%sf(j,k,l) * &
                                                       grad_z_vf(We_idx(i,2))%sf(j,k,l) / &
                                                     MAX(norm_vf(We_idx(i,2))%sf(j,k,l),sgm_eps)
    
                                    tau_We_vf(mom_idx%end)%sf(j,k,l) = &
                                    tau_We_vf(mom_idx%end)%sf(j,k,l) - &
                                    tau_We(1,3)
    
                                    tau_We_vf(E_idx)%sf(j,k,l) = &
                                    tau_We_vf(E_idx)%sf(j,k,l) - &
                                    q_prim_vf(mom_idx%end)%sf(j,k,l) * tau_We(1,3)
    
                                END DO
                            END DO
                        END DO
                    END DO
                END IF

            ! End capillary stresses in x-direction =============================

            ! Capillary stresses in y-direction =============================
            ELSEIF (norm_dir == 2) THEN
                DO i = 1, We_size
                    DO l = iz%beg, iz%end
                        DO k = iy%beg, iy%end
                            DO j = ix%beg, ix%end

                                CALL s_convert_to_mixture_variables(   q_prim_vf, rho_cap, &
                                                                    gamma_cap, pi_inf_cap, &
                                                                    Re_cap, We_cap, j,k,l  )

                                tau_We(2,1) = -1d0/ We_cap(We_idx(i,1),We_idx(i,2)) * &
                                                   grad_y_vf(We_idx(i,2))%sf(j,k,l) * &
                                                   grad_x_vf(We_idx(i,2))%sf(j,k,l) / &
                                                 MAX(norm_vf(We_idx(i,2))%sf(j,k,l),sgm_eps)

                                tau_We(2,2) =  1d0/ We_cap(We_idx(i,1),We_idx(i,2)) * &
                                                   ( norm_vf(We_idx(i,2))%sf(j,k,l) - &
                                                   grad_y_vf(We_idx(i,2))%sf(j,k,l) * &
                                                   grad_y_vf(We_idx(i,2))%sf(j,k,l) / &
                                                 MAX(norm_vf(We_idx(i,2))%sf(j,k,l),sgm_eps))

                                DO r = 1,2
                                    tau_We_vf(cont_idx%end+r)%sf(j,k,l) = &
                                    tau_We_vf(cont_idx%end+r)%sf(j,k,l) - &
                                    tau_We(2,r)
                                END DO

                                tau_We(2,2) = -1d0/ We_cap(We_idx(i,1),We_idx(i,2)) * &
                                                   grad_y_vf(We_idx(i,2))%sf(j,k,l) * &
                                                   grad_y_vf(We_idx(i,2))%sf(j,k,l) / &
                                                 MAX(norm_vf(We_idx(i,2))%sf(j,k,l),sgm_eps)

                                DO r = 1,2
                                    tau_We_vf(E_idx)%sf(j,k,l) = &
                                    tau_We_vf(E_idx)%sf(j,k,l) - &
                                    q_prim_vf(cont_idx%end+r)%sf(j,k,l) * tau_We(2,r)
                                END DO

                            END DO
                        END DO
                    END DO
                END DO

                IF (p > 0) THEN
                    DO i = 1, We_size
                        DO l = iz%beg, iz%end
                            DO k = iy%beg, iy%end
                                DO j = ix%beg, ix%end
    
                                    CALL s_convert_to_mixture_variables(   q_prim_vf, rho_cap, &
                                                                        gamma_cap, pi_inf_cap, &
                                                                        Re_cap, We_cap, j,k,l  )
    
                                    tau_We(2,3) = -1d0/ We_cap(We_idx(i,1),We_idx(i,2)) * &
                                                       grad_y_vf(We_idx(i,2))%sf(j,k,l) * &
                                                       grad_z_vf(We_idx(i,2))%sf(j,k,l) / &
                                                     MAX(norm_vf(We_idx(i,2))%sf(j,k,l),sgm_eps)
    
                                    tau_We_vf(mom_idx%end)%sf(j,k,l) = &
                                    tau_We_vf(mom_idx%end)%sf(j,k,l) - &
                                    tau_We(2,3)
    
                                    tau_We_vf(E_idx)%sf(j,k,l) = &
                                    tau_We_vf(E_idx)%sf(j,k,l) - &
                                    q_prim_vf(mom_idx%end)%sf(j,k,l) * tau_We(2,3)
    
                                END DO
                            END DO
                        END DO
                    END DO
                END IF

            ! End capillary stresses in y-direction =============================

            ! Capillary stresses in z-direction =============================
            ELSE
                DO i = 1, We_size
                    DO l = iz%beg, iz%end
                        DO k = iy%beg, iy%end
                            DO j = ix%beg, ix%end

                                tau_We(3,1) = -1d0/ We_cap(We_idx(i,1),We_idx(i,2)) * &
                                                   grad_z_vf(We_idx(i,2))%sf(j,k,l) * &
                                                   grad_x_vf(We_idx(i,2))%sf(j,k,l) / &
                                                 MAX(norm_vf(We_idx(i,2))%sf(j,k,l),sgm_eps)

                                tau_We(3,2) = -1d0/ We_cap(We_idx(i,1),We_idx(i,2)) * &
                                                   grad_z_vf(We_idx(i,2))%sf(j,k,l) * &
                                                   grad_y_vf(We_idx(i,2))%sf(j,k,l) / &
                                                 MAX(norm_vf(We_idx(i,2))%sf(j,k,l),sgm_eps)

                                tau_We(3,3) =  1d0/ We_cap(We_idx(i,1),We_idx(i,2)) * &
                                                   ( norm_vf(We_idx(i,2))%sf(j,k,l) - &
                                                   grad_z_vf(We_idx(i,2))%sf(j,k,l) * &
                                                   grad_z_vf(We_idx(i,2))%sf(j,k,l) / &
                                                 MAX(norm_vf(We_idx(i,2))%sf(j,k,l),sgm_eps))

                                DO r = 1,3
                                    tau_We_vf(cont_idx%end+r)%sf(j,k,l) = &
                                    tau_We_vf(cont_idx%end+r)%sf(j,k,l) - &
                                    tau_We(3,r)
                                END DO

                                tau_We(3,3) = -1d0/ We_cap(We_idx(i,1),We_idx(i,2)) * &
                                                   grad_z_vf(We_idx(i,2))%sf(j,k,l) * &
                                                   grad_z_vf(We_idx(i,2))%sf(j,k,l) / &
                                                 MAX(norm_vf(We_idx(i,2))%sf(j,k,l),sgm_eps)

                                DO r = 1,3
                                    tau_We_vf(E_idx)%sf(j,k,l) = &
                                    tau_We_vf(E_idx)%sf(j,k,l) - &
                                    q_prim_vf(cont_idx%end+r)%sf(j,k,l) * tau_We(3,r)
                                END DO

                            END DO
                        END DO
                    END DO
                END DO

            END IF
            ! End capillary stresses in z-direction =============================

        END SUBROUTINE s_compute_capillary_stress_tensor ! ---------------------------------





        SUBROUTINE s_compute_capillary_source(i,q_prim_vf) ! ------------------------------
        ! Description: The purpose of this procedure is to compute the source term
        !              that accounts for capillary effects in the momentum equations

            ! Dimensional split index
            INTEGER, INTENT(IN) :: i

            ! Cell-average primitive variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: q_prim_vf

            ! Mixture variables
            REAL(KIND(0d0)) :: rho_cap, gamma_cap, pi_inf_cap
            REAL(KIND(0d0)), DIMENSION(2) :: Re_cap
            REAL(KIND(0d0)), DIMENSION(1:num_fluids,1:num_fluids) :: We_cap
            REAL(KIND(0d0)), DIMENSION(1:num_fluids) :: kappa_cap

            REAL(KIND(0d0)) :: F_sv

            ! Generic loop iterators
            INTEGER :: j,k,l,r

            DO j = 0, m
                DO k = 0, n
                    DO l = 0, p
                        DO r = 1, crv_size
                            IF (alt_crv .NEQV. .TRUE.) THEN
                                kappa_cap(crv_idx(r)) = kappa_vf(crv_idx(r))%sf(j,k,l)
                            ELSE
                                kappa_cap(crv_idx(r)) = alt_kappa_vf(crv_idx(r))%sf(j,k,l)
                            END IF
                        END DO
                        CALL s_convert_to_mixture_variables(   q_prim_vf, rho_cap, &
                                                            gamma_cap, pi_inf_cap, &
                                                            Re_cap, We_cap, j,k,l  )
                        We_mtm_src(j,k,l) = 0d0
                        We_nrg_src(j,k,l) = 0d0

                        DO r = 1, We_size
                            IF (i == 1) THEN 
                                F_sv = -1d0/We_cap(We_idx(r,1),We_idx(r,2)) * &
                                                     kappa_cap(We_idx(r,2)) * &
                                dq_prim_dx_qp(0,0,0)%vf(adv_idx%beg)%sf(j,k,l)
                            ELSEIF (i == 2) THEN
                                F_sv = -1d0/We_cap(We_idx(r,1),We_idx(r,2)) * &
                                                     kappa_cap(We_idx(r,2)) * &
                                dq_prim_dy_qp(0,0,0)%vf(adv_idx%beg)%sf(j,k,l)
                            ELSEIF (i == 3) THEN
                                F_sv = -1d0/We_cap(We_idx(r,1),We_idx(r,2)) * &
                                                     kappa_cap(We_idx(r,2)) * &
                                dq_prim_dz_qp(0,0,0)%vf(adv_idx%beg)%sf(j,k,l)
                            END IF

                            We_mtm_src(j,k,l) = We_mtm_src(j,k,l) + F_sv
                            We_nrg_src(j,k,l) = We_nrg_src(j,k,l) + F_sv * &
                                        q_prim_vf(cont_idx%end+i)%sf(j,k,l)
                        END DO
                    END DO
                END DO
            END DO
        END SUBROUTINE s_compute_capillary_source ! ----------------------------------

        SUBROUTINE s_get_divergence(idir,q_prim_vf,mydivu)
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: q_prim_vf
            TYPE(scalar_field), intent(inout) :: mydivu
            INTEGER, INTENT(IN) :: idir
            INTEGER :: j,k,l
            
            !contribute to divergence computation \div(u)
            if (idir == 1) mydivu%sf(:,:,:) = 0d0
            
            do j = 0,m
                do k = 0,n
                    do l = 0,p
                        if (idir == 1) then
                            mydivu%sf(j,k,l) =                      0.5d0/dx(j)*(q_prim_vf(cont_idx%end+idir)%sf(j+1,k,l) - &
                                                                                 q_prim_vf(cont_idx%end+idir)%sf(j-1, k , l )) 
                        else if (idir == 2) then
                            mydivu%sf(j,k,l) = mydivu%sf(j,k,l) +   0.5d0/dy(k)*(q_prim_vf(cont_idx%end+idir)%sf(j,k+1,l) - &
                                                                                 q_prim_vf(cont_idx%end+idir)%sf( j ,k-1, l ))
                        else if (idir == 3) then
                            mydivu%sf(j,k,l) = mydivu%sf(j,k,l) +   0.5d0/dz(l)*(q_prim_vf(cont_idx%end+idir)%sf(j,k,l+1) - &
                                                                                 q_prim_vf(cont_idx%end+idir)%sf( j , k ,l-1))
                        end if
                    end do
                end do
            end do

        END SUBROUTINE s_get_divergence


        SUBROUTINE s_get_monopole(idir, q_prim_vf,t_step,mymono) ! ------------------------------
        ! Description: The purpose of this procedure is to compute the source term
        ! that are needed for generating one-way acoustic waves

            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: q_prim_vf
            TYPE(mono_parameters), INTENT(IN) :: mymono
            INTEGER, INTENT(IN) :: idir, t_step
            INTEGER :: ndirs,j,k,l
            real(kind(0d0)) :: mytime, sound, n_tait, B_tait
            real(kind(0d0)) :: s2, myRho, const_sos
            
            REAL(KIND(0d0)), DIMENSION(2) :: Re
            REAL(KIND(0d0)), DIMENSION( num_fluids, &
                                        num_fluids  ) :: We
           
            ndirs = 1; if (n > 0) ndirs = 2; if (p > 0) ndirs = 3



            if (idir == ndirs) then
                mytime = t_step*dt

                do j = 0,m; do k = 0,n; do l=0,p
                    CALL s_convert_to_mixture_variables( q_prim_vf, myRho, n_tait, B_tait, Re, We, j, k, l )
                    n_tait = 1.d0/n_tait + 1.d0 !make this the usual little 'gamma'

                    sound = n_tait*(q_prim_vf(E_idx)%sf(j,k,l) + ((n_tait-1d0)/n_tait)*B_tait)/myRho
                    sound = dsqrt(sound)

                    const_sos = dsqrt( n_tait )

                    !c_ac=SQRT((1d0/gamma_ac+1d0)*((q_prim_vf(E_idx)%sf(j,k,l)+1d0/(gamma_ac+1d0)*pi_inf_ac))/rho_ac)

                    s2 = f_g(mytime,sound,const_sos,mymono) * f_delta(j,k,l,mymono%loc,mymono%length,mymono) !g(t)*\delta(x-x0)
                   
                    mono_mass_src(j,k,l)    = mono_mass_src(j,k,l) + s2/sound
                    if (n ==0) then
                        
                        !1d
                        if (mymono%dir < -0.1d0) then
                            !left-going wave
                            mono_mom_src(1,j,k,l) = mono_mom_src(1,j,k,l) - s2
                        else
                            !right-going wave
                            mono_mom_src(1,j,k,l) = mono_mom_src(1,j,k,l) + s2
                        end if
                    else if (p==0) then
                        if ( (j==1) .and. (k==1) .and. proc_rank == 0) &
                            print*, '====== Monopole magnitude: ', f_g(mytime,sound,const_sos,mymono) 
 
                        if (mymono%dir .ne. dflt_real) then
                            !2d
                            !mono_mom_src(1,j,k,l) = s2
                            !mono_mom_src(2,j,k,l) = s2
                            mono_mom_src(1,j,k,l) = mono_mom_src(1,j,k,l) + s2*cos( mymono%dir )
                            mono_mom_src(2,j,k,l) = mono_mom_src(2,j,k,l) + s2*sin( mymono%dir )
                        end if
                    else    
                        !3Di
                        if (mymono%dir .ne. dflt_real) then
                            mono_mom_src(1,j,k,l) = mono_mom_src(1,j,k,l) + s2*cos( mymono%dir )
                            mono_mom_src(2,j,k,l) = mono_mom_src(2,j,k,l) + s2*sin( mymono%dir )
                        end if
                    end if

                    if (model_eqns .ne. 4) then
                        mono_E_src(j,k,l)   = mono_E_src(j,k,l) + s2*sound/(n_tait - 1.d0)
                    end if
                end do; end do; end do
            end if

        END SUBROUTINE s_get_monopole
       

        function f_g(mytime,sos,mysos,mymono)
            real(kind(0d0)), intent(in) :: mytime, sos, mysos
            type(mono_parameters), intent(in) :: mymono
            real(kind(0d0)) :: period, t0, sigt, pa
            real(kind(0d0)) :: f_g

            if (mymono%pulse == 1) then
                !sine wave
                period = mymono%length/sos
                if (mytime .le. mymono%npulse*period) then
                    f_g = mymono%mag*sin(mytime*2.d0*pi/period)
                else
                    f_g = 0d0
                end if
            else if (mymono%pulse == 2) then
                !Gaussian pulse
                sigt = mymono%length/sos/7.d0
                t0 = 3.5d0*sigt 
                f_g = mymono%mag/(dsqrt(2.d0*pi) * sigt) * &
                    dexp( -0.5d0 * ((mytime-t0)**2.d0)/(sigt**2.d0) )
            else if (mymono%pulse == 3) then
                !square wave
                sigt = mymono%length/sos
                t0 = 0d0
                if (mytime > t0 .and. mytime < sigt) then
                    f_g = mymono%mag
                else 
                    f_g = 0d0
                end if
            else
                PRINT '(A)', 'No pulse type detected. Exiting ...'
                CALL s_mpi_abort()
            end if

        end function f_g

        function f_delta(j,k,l,mono_loc,mono_leng,mymono)
            !spatial delta function
            real(kind(0d0)), dimension(3), intent(in) :: mono_loc
            type(mono_parameters), intent(in) :: mymono
            real(kind(0d0)), intent(in) :: mono_leng
            integer, intent(in) :: j,k,l

            integer :: q
            real(kind(0d0)) :: h,hx,hy,hz
            real(kind(0d0)) :: hxnew,hynew
            real(kind(0d0)) :: sig
            real(kind(0d0)) :: f_delta

            if (n==0) then
                sig = dx(j)
                sig = sig*2.5d0
            else if (p==0) then
                sig = maxval( (/ dx(j), dy(k) /) )
                sig = sig*2.5d0
            else 
                sig = maxval( (/ dx(j), dy(k), dz(l) /) )
                sig = sig*2.5d0
            end if            

            if (n==0) then      !1D
                if (mymono%support == 1) then
                    !1d delta function
                    hx = abs(mono_loc(1) - x_cc(j))

                    f_delta = 1.d0/(dsqrt(2.d0*pi)*sig/2.d0) * &
                        dexp( -0.5d0 * (hx/(sig/2.d0))**2.d0 )
                else if (mymono%support == 0) then
                    !support for all x
                    f_delta = 1.d0
                end if
            else if (p==0) then !2D
                hx = mono_loc(1) - x_cc(j)
                hy = mono_loc(2) - y_cc(k)
                if (mymono%support == 1) then
                    !2d delta function
                    sig = mono_leng/20.d0
                    h = dsqrt(hx**2.d0 + hy**2.d0)

                    f_delta = 1.d0/(dsqrt(2.d0*pi)*sig/2.d0) * &
                        dexp( -0.5d0 * ((h/(sig/2.d0))**2.d0) )
                else if (mymono%support == 2) then
                    !only support for y \pm some value
                    if ( abs(hy) < mymono%length ) then
                        f_delta = 1.d0/(dsqrt(2.d0*pi)*sig/2.d0) * &
                            dexp( -0.5d0 * (hx/(sig/2.d0))**2.d0 )
                    else
                        f_delta = 0d0
                    end if
                else if (mymono%support == 3) then
                    !only support along some line

                    hx = x_cc(j) - mono_loc(1) 
                    hy = y_cc(k) - mono_loc(2)
                    !rotate actual point by -theta
                    hxnew = cos(      mymono%dir)*hx + sin(     mymono%dir)*hy
                    hynew = -1.d0*sin(mymono%dir)*hx + cos(     mymono%dir)*hy
                    !if ( abs(hynew) < mymono%length*2 ) then
                    if ( abs(hynew) < mymono%loc(3)/2.d0 ) then
                        f_delta = 1.d0/(dsqrt(2.d0*pi)*sig/2.d0) * &
                            dexp( -0.5d0 * (hxnew/(sig/2.d0))**2.d0 )
                    else
                        f_delta = 0d0
                    end if
                else if (mymono%support == 4) then
                    !support for all y
                    f_delta = 1.d0/(dsqrt(2.d0*pi)*sig) * &
                        dexp( -0.5d0 * (hx/sig)**2.d0 ) 
                end if
            else !3D
                if (mymono%support == 3) then
                    !only support along some line

                    hx = x_cc(j) - mono_loc(1) 
                    hy = y_cc(k) - mono_loc(2)
                    hz = z_cc(l) - mono_loc(3)
                    !rotate actual point by -theta
                    hxnew = cos(      mymono%dir)*hx + sin(     mymono%dir)*hy
                    hynew = -1.d0*sin(mymono%dir)*hx + cos(     mymono%dir)*hy
                    !if ( abs(hynew) < mymono%length*2 ) then
                    if ( abs(hynew) < mymono%length/2. .and. &
                         abs(hz) < mymono%length/2. ) then
                        f_delta = 1.d0/(dsqrt(2.d0*pi)*sig/2.d0) * &
                            dexp( -0.5d0 * (hxnew/(sig/2.d0))**2.d0 )
                    else
                        f_delta = 0d0
                    end if
                else
                    print '(a)', 'monopole support not properly defined'
                    call s_mpi_abort()
                end if
            end if
 
        end function f_delta

   
        SUBROUTINE s_compute_regularization_source(i,q_prim_vf) ! -----------------
        ! Description: The purpose of this procedure is to compute the interface
        !               sharpening regularization source terms. Only applicable
        !               for 2-fluid system!

            ! Dimensional split index
            INTEGER, INTENT(IN) :: i

            ! Cell-average primitive variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(IN) :: q_prim_vf

            TYPE(scalar_field), ALLOCATABLE :: var
            TYPE(scalar_field), ALLOCATABLE :: grad_x, grad_y, grad_z
            TYPE(scalar_field), ALLOCATABLE :: alpharho_grad_x, alpharho_grad_y, alpharho_grad_z
            TYPE(scalar_field), ALLOCATABLE :: norm
            TYPE(scalar_field), ALLOCATABLE :: un_alpha_x, un_alpha_y, un_alpha_z

            REAL(KIND(0d0)), DIMENSION(0:m,0:n,0:p) :: Lheaviside, U0, velmag
            REAL(KIND(0d0)) :: U0_loc, U0_glb
            REAL(KIND(0d0)), DIMENSION(0:m,0:n,0:p) :: Rnohat, R1hat, R2hat
            REAL(KIND(0d0)), DIMENSION(num_dims) :: vel

            TYPE(bounds_info) :: ix,iy,iz
            ! Generic loop iterators
            INTEGER :: j,k,l,r

            ix%beg = -buff_size; iy%beg = -buff_size
            ix%end = m + buff_size; iy%end = n + buff_size
            IF (p > 0) THEN
                iz%beg = -buff_size; iz%end = p + buff_size
            ELSE
                iz%beg = 0; iz%end = 0
            END IF
            ALLOCATE(            var%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
            ALLOCATE(         grad_x%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
            ALLOCATE(         grad_y%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
            ALLOCATE(         grad_z%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
            ALLOCATE(alpharho_grad_x%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
            ALLOCATE(alpharho_grad_y%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
            ALLOCATE(alpharho_grad_z%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
            ALLOCATE(           norm%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
            ALLOCATE(     un_alpha_x%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
            ALLOCATE(     un_alpha_y%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
            ALLOCATE(     un_alpha_z%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))

            DO j = 0, m
                DO k = 0, n
                    DO l = 0, p
                        IF ((q_prim_vf(adv_idx%beg)%sf(j,k,l) > 1d-6) &
                                        .AND.                         &
                            (q_prim_vf(adv_idx%beg)%sf(j,k,l) < (1d0-1d-6))) THEN
                            Lheaviside(j,k,l) = 1d0
                        ELSE
                            Lheaviside(j,k,l) = 0d0
                        END IF

                        DO r = 1, num_dims
                            vel(r) = q_prim_vf(cont_idx%end+r)%sf(j,k,l)
                        END DO

                        velmag(j,k,l) = SQRT(DOT_PRODUCT(vel,vel))

                        U0(j,k,l) = 4d0*q_prim_vf(adv_idx%beg)%sf(j,k,l)   * &
                                    (1d0-q_prim_vf(adv_idx%beg)%sf(j,k,l)) * &
                                    velmag(j,k,l)
                    END DO
                END DO
            END DO

            U0_loc = MAXVAL(U0)
            IF (num_procs > 1) THEN
                CALL s_mpi_allreduce_max(U0_loc,U0_glb)
            ELSE
                U0_glb = U0_loc
            END IF

            var%sf(:,:,:) = q_prim_vf(adv_idx%beg)%sf(:,:,:)
            CALL s_compute_fd_gradient(var,grad_x,grad_y,grad_z,norm)
            un_alpha_x%sf(:,:,:) = grad_x%sf(:,:,:)/MAX(norm%sf(:,:,:),sgm_eps)
            un_alpha_y%sf(:,:,:) = grad_y%sf(:,:,:)/MAX(norm%sf(:,:,:),sgm_eps)
            un_alpha_z%sf(:,:,:) = grad_z%sf(:,:,:)/MAX(norm%sf(:,:,:),sgm_eps)

            DO j = ix%beg, ix%end
                DO k = iy%beg, iy%end
                    DO l = iz%beg, iz%end
                        var%sf(j,k,l) = reg_eps*norm%sf(j,k,l) - q_prim_vf(adv_idx%beg)%sf(j,k,l) * &
                                                        (1d0 - q_prim_vf(adv_idx%beg)%sf(j,k,l))
                    END DO
                END DO
            END DO
            CALL s_compute_fd_gradient(var,grad_x,grad_y,grad_z,norm)
            DO j = 0, m
                DO k = 0, n
                    DO l = 0, p
                        IF (p > 0) THEN
                            Rnohat(j,k,l) = Lheaviside(j,k,l)*U0_glb * &
                                            (un_alpha_x%sf(j,k,l)*grad_x%sf(j,k,l) + &
                                             un_alpha_y%sf(j,k,l)*grad_y%sf(j,k,l) + &
                                             un_alpha_z%sf(j,k,l)*grad_z%sf(j,k,l))
                        ELSE
                            Rnohat(j,k,l) = Lheaviside(j,k,l)*U0_glb * &
                                            (un_alpha_x%sf(j,k,l)*grad_x%sf(j,k,l) + &
                                             un_alpha_y%sf(j,k,l)*grad_y%sf(j,k,l))
                        END IF
                    END DO
                END DO
            END DO
            
            DO r = cont_idx%beg, cont_idx%end
                var%sf(:,:,:) = q_prim_vf(r)%sf(:,:,:)
                CALL s_compute_fd_gradient(var,alpharho_grad_x,alpharho_grad_y,alpharho_grad_z,norm)
                DO j = ix%beg, ix%end
                    DO k = iy%beg, iy%end
                        DO l = iz%beg, iz%end
                            IF (p > 0) THEN
                                var%sf(j,k,l) = reg_eps * &
                                                (un_alpha_x%sf(j,k,l)*alpharho_grad_x%sf(j,k,l) + &
                                                 un_alpha_y%sf(j,k,l)*alpharho_grad_y%sf(j,k,l) + &
                                                 un_alpha_z%sf(j,k,l)*alpharho_grad_z%sf(j,k,l))
                            ELSE
                                var%sf(j,k,l) = reg_eps * &
                                                (un_alpha_x%sf(j,k,l)*alpharho_grad_x%sf(j,k,l) + &
                                                 un_alpha_y%sf(j,k,l)*alpharho_grad_y%sf(j,k,l))
                            END IF
                        END DO
                    END DO
                END DO
                CALL s_compute_fd_gradient(var,grad_x,grad_y,grad_z,norm)
                DO j = 0, m
                    DO k = 0, n
                        DO l = 0, p
                            IF (p > 0) THEN
                                var%sf(j,k,l) = Lheaviside(j,k,l)*U0_glb * &
                                               (un_alpha_x%sf(j,k,l)*(grad_x%sf(j,k,l) - (1d0-2d0*q_prim_vf(adv_idx%beg)%sf(j,k,l))*alpharho_grad_x%sf(j,k,l)) + &
                                                un_alpha_y%sf(j,k,l)*(grad_y%sf(j,k,l) - (1d0-2d0*q_prim_vf(adv_idx%beg)%sf(j,k,l))*alpharho_grad_y%sf(j,k,l)) + &
                                                un_alpha_z%sf(j,k,l)*(grad_z%sf(j,k,l) - (1d0-2d0*q_prim_vf(adv_idx%beg)%sf(j,k,l))*alpharho_grad_z%sf(j,k,l)))
                            ELSE
                                var%sf(j,k,l) = Lheaviside(j,k,l)*U0_glb * &
                                               (un_alpha_x%sf(j,k,l)*(grad_x%sf(j,k,l) - (1d0-2d0*q_prim_vf(adv_idx%beg)%sf(j,k,l))*alpharho_grad_x%sf(j,k,l)) + &
                                                un_alpha_y%sf(j,k,l)*(grad_y%sf(j,k,l) - (1d0-2d0*q_prim_vf(adv_idx%beg)%sf(j,k,l))*alpharho_grad_y%sf(j,k,l))) 
                            END IF
                        END DO
                    END DO
                END DO
                IF (r == cont_idx%beg) THEN
                    R1hat(:,:,:) = var%sf(0:m,0:n,0:p)
                ELSEIF (r == cont_idx%end) THEN
                    R2hat(:,:,:) = var%sf(0:m,0:n,0:p)
                END IF
            END DO

            reg_src_vf(cont_idx%beg)%sf(:,:,:) = R1hat(:,:,:)
            reg_src_vf(cont_idx%end)%sf(:,:,:) = R2hat(:,:,:)
            DO r = mom_idx%beg, mom_idx%end
                reg_src_vf(r)%sf(:,:,:) = q_prim_vf(r)%sf(:,:,:)*(R1hat(:,:,:) + R2hat(:,:,:))
            END DO
            reg_src_vf(E_idx)%sf(:,:,:) = 5d-1*velmag(:,:,:)**2d0*(R1hat(:,:,:) + R2hat(:,:,:)) + &
                                            (q_prim_vf(E_idx)%sf(:,:,:)*(fluid_pp(1)%gamma - fluid_pp(2)%gamma) + &
                                            fluid_pp(1)%pi_inf - fluid_pp(2)%pi_inf)*Rnohat(:,:,:)
            reg_src_vf(adv_idx%beg)%sf(:,:,:) = Rnohat(:,:,:)
            IF (adv_alphan) THEN
                reg_src_vf(adv_idx%end)%sf(:,:,:) = -Rnohat(:,:,:)
            END IF

            DEALLOCATE(var%sf, grad_x%sf, grad_y%sf, grad_z%sf, norm%sf)
            DEALLOCATE(un_alpha_x%sf, un_alpha_y%sf, un_alpha_z%sf)
            DEALLOCATE(alpharho_grad_x%sf, alpharho_grad_y%sf, alpharho_grad_z%sf)

        END SUBROUTINE s_compute_regularization_source ! ----------------------------------





        SUBROUTINE s_compute_fd_gradient(var,grad_x,grad_y,grad_z,norm)

            TYPE(scalar_field), INTENT(IN) :: var
            TYPE(scalar_field), INTENT(INOUT) :: grad_x
            TYPE(scalar_field), INTENT(INOUT) :: grad_y
            TYPE(scalar_field), INTENT(INOUT) :: grad_z
            TYPE(scalar_field), INTENT(INOUT) :: norm

            TYPE(bounds_info) :: ix,iy,iz
            ! Generic loop iterators
            INTEGER :: j,k,l

            ix%beg = -buff_size; ix%end = m + buff_size;
            IF (n > 0) THEN
                iy%beg = -buff_size; iy%end = n + buff_size
                IF (p > 0) THEN
                    iz%beg = -buff_size; iz%end = p + buff_size
                ELSE
                    iz%beg = -1; iz%end = 1
                END IF
            ELSE
                iy%beg = -1; iy%end = 1
            END IF

            DO j = ix%beg+1, ix%end-1
                DO k = iy%beg+1, iy%end-1
                    DO l = iz%beg+1, iz%end-1
                        grad_x%sf(j,k,l) = (var%sf(j+1,k,l) - var%sf(j-1,k,l))/(x_cc(j+1) - x_cc(j-1))
                        IF (n > 0) THEN
                            grad_y%sf(j,k,l) = (var%sf(j,k+1,l) - var%sf(j,k-1,l))/(y_cc(k+1) - y_cc(k-1))
                            IF (p > 0) THEN
                                grad_z%sf(j,k,l) = (var%sf(j,k,l+1) - var%sf(j,k,l-1))/(z_cc(l+1) - z_cc(l-1))
                            END IF
                        END IF
                    END DO
                END DO
            END DO
            grad_x%sf(ix%beg,:,:) = (-3d0*var%sf(ix%beg,:,:) + 4d0*var%sf(ix%beg+1,:,:) - var%sf(ix%beg+2,:,:)) / &
                                    (x_cc(ix%beg+2) - x_cc(ix%beg))
            grad_x%sf(ix%end,:,:) = ( 3d0*var%sf(ix%end,:,:) - 4d0*var%sf(ix%end-1,:,:) + var%sf(ix%end-2,:,:)) / &
                                    (x_cc(ix%end) - x_cc(ix%end-2))
            IF (n > 0) THEN
                grad_y%sf(:,iy%beg,:) = (-3d0*var%sf(:,iy%beg,:) + 4d0*var%sf(:,iy%beg+1,:) - var%sf(:,iy%beg+2,:)) / &
                                        (y_cc(iy%beg+2) - y_cc(iy%beg))
                grad_y%sf(:,iy%end,:) = ( 3d0*var%sf(:,iy%end,:) - 4d0*var%sf(:,iy%end-1,:) + var%sf(:,iy%end-2,:)) / &
                                        (y_cc(iy%end) - y_cc(iy%end-2))
                IF (p > 0) THEN
                    grad_z%sf(:,:,iz%beg) = (-3d0*var%sf(:,:,iz%beg) + 4d0*var%sf(:,:,iz%beg+1) - var%sf(:,:,iz%beg+2)) / &
                                            (z_cc(iz%beg+2) - z_cc(iz%beg))
                    grad_z%sf(:,:,iz%end) = ( 3d0*var%sf(:,:,iz%end) - 4d0*var%sf(:,:,iz%end-1) + var%sf(:,:,iz%end-2)) / &
                                            (z_cc(iz%end) - z_cc(iz%end-2))
                END IF
            END IF

            IF (bc_x%beg <= -3) THEN
                grad_x%sf(0,:,:) = (-3d0*var%sf(0,:,:) + 4d0*var%sf(1,:,:) - var%sf(2,:,:)) / &
                                   (x_cc(2) - x_cc(0))
            END IF
            IF (bc_x%end <= -3) THEN
                grad_x%sf(m,:,:) = ( 3d0*var%sf(m,:,:) - 4d0*var%sf(m-1,:,:) + var%sf(m-2,:,:)) / &
                                   (x_cc(m) - x_cc(m-2))
            END IF
            IF (n > 0) THEN
                IF (bc_y%beg <= -3 .AND. bc_y%beg /= -13) THEN
                    grad_y%sf(:,0,:) = (-3d0*var%sf(:,0,:) + 4d0*var%sf(:,1,:) - var%sf(:,2,:)) / &
                                       (y_cc(2) - y_cc(0))
                END IF
                IF (bc_y%end <= -3) THEN
                    grad_y%sf(:,n,:) = ( 3d0*var%sf(:,n,:) - 4d0*var%sf(:,n-1,:) + var%sf(:,n-2,:)) / &
                                       (y_cc(n) - y_cc(n-2))
                END IF
                IF (p > 0) THEN
                    IF (bc_z%beg <= -3) THEN
                        grad_z%sf(:,:,0) = (-3d0*var%sf(:,:,0) + 4d0*var%sf(:,:,1) - var%sf(:,:,2)) / &
                                           (z_cc(2) - z_cc(0))
                    END IF
                    IF (bc_z%end <= -3) THEN
                        grad_z%sf(:,:,p) = ( 3d0*var%sf(:,:,p) - 4d0*var%sf(:,:,p-1) + var%sf(:,:,p-2)) / &
                                           (z_cc(p) - z_cc(p-2))
                    END IF
                END IF
            END IF

            IF (p == 0) THEN
                iz%beg = 0; iz%end = 0
                IF (n == 0) THEN
                    iy%beg = 0; iy%end = 0
                END IF
            END IF
            DO j = ix%beg, ix%end
                DO k = iy%beg, iy%end
                    DO l = iz%beg, iz%end
                        IF (p > 0) THEN
                            norm%sf(j,k,l) = SQRT(grad_x%sf(j,k,l)**2d0 + &
                                                  grad_y%sf(j,k,l)**2d0 + &
                                                  grad_z%sf(j,k,l)**2d0)
                        ELSEIF (n > 0) THEN
                            norm%sf(j,k,l) = SQRT(grad_x%sf(j,k,l)**2d0 + &
                                                  grad_y%sf(j,k,l)**2d0)
                        ELSE
                            norm%sf(j,k,l) = grad_x%sf(j,k,l)
                        END IF
                    END DO
                END DO
            END DO

        END SUBROUTINE s_compute_fd_gradient ! --------------------------------------




        SUBROUTINE s_pressure_relaxation_procedure(q_cons_vf) ! ----------------
        ! Description: The purpose of this procedure is to infinitely relax
        !              the pressures from the internal-energy equations to a
        !              unique pressure, from which the corresponding volume
        !              fraction of each phase are recomputed. For conservation
        !              purpose, this pressure is finally corrected using the
        !              mixture-total-energy equation.
            
            ! Cell-average conservative variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: q_cons_vf
            
            ! Relaxed pressure, initial partial pressures, function f(p) and its partial
            ! derivative df(p), isentropic partial density, sum of volume fractions,
            ! mixture density, dynamic pressure, surface energy, specific heat ratio
            ! function, liquid stiffness function (two variations of the last two
            ! ones), shear and volume Reynolds numbers and the Weber numbers
            REAL(KIND(0d0))                                   ::  pres_relax
            REAL(KIND(0d0)), DIMENSION(num_fluids)            :: pres_K_init
            REAL(KIND(0d0))                                   ::      f_pres
            REAL(KIND(0d0))                                   ::     df_pres
            REAL(KIND(0d0)), DIMENSION(num_fluids)            ::     rho_K_s
            REAL(KIND(0d0))                                   ::   sum_alpha
            REAL(KIND(0d0))                                   ::         rho
            REAL(KIND(0d0))                                   ::    dyn_pres
            REAL(KIND(0d0))                                   ::        E_We
            REAL(KIND(0d0))                                   ::       gamma
            REAL(KIND(0d0))                                   ::      pi_inf
            REAL(KIND(0d0)), DIMENSION(num_fluids)            ::   gamma_min
            REAL(KIND(0d0)), DIMENSION(num_fluids)            ::    pres_inf
            REAL(KIND(0d0)), DIMENSION(2)                     ::          Re
            REAL(KIND(0d0)), DIMENSION(num_fluids,num_fluids) ::          We

            ! Generic loop iterators
            INTEGER :: i,j,k,l,iter

            ! Relaxation procedure determination variable
            INTEGER :: relax
            
            DO i = 1, num_fluids
                gamma_min(i) = 1d0/fluid_pp(i)%gamma + 1d0
                pres_inf(i)  = fluid_pp(i)%pi_inf / (1d0+fluid_pp(i)%gamma)
            END DO

            DO j = 0, m
                DO k = 0, n
                    DO l = 0, p

                        ! Numerical correction of the volume fractions
                        IF (mpp_lim) THEN
                            sum_alpha = 0d0
                            DO i = 1, num_fluids
                                IF ((q_cons_vf(i+cont_idx%beg-1)%sf(j,k,l) .LT. 0d0) .OR. (q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l) .LT. 0d0)) THEN
                                    q_cons_vf(i+cont_idx%beg-1)%sf(j,k,l) = 0d0
                                    q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l)  = 0d0
                                    q_cons_vf(i+internalEnergies_idx%beg-1)%sf(j,k,l)  = 0d0
                                END IF
                                IF (q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l) .GT. 1d0) q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l) = 1d0
                                sum_alpha = sum_alpha + q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l)
                            END DO
                            DO i = 1, num_fluids
                                q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l) = q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l) / sum_alpha
                            END DO
                        END IF


                        ! Pressures relaxation procedure ===================================

                        ! Is the pressure relaxation procedure necessary?
                        relax = 1
                        DO i = 1, num_fluids
                            IF (q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l) .GT. (1d0-sgm_eps)) relax = 0
                        END DO

                        IF (relax == 1) THEN
                            ! Initial state
                            pres_relax = 0d0
                            DO i = 1, num_fluids
                                IF (q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l) .GT. sgm_eps) THEN
                                    pres_K_init(i) = (q_cons_vf(i+internalEnergies_idx%beg-1)%sf(j,k,l) / q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l) &
                                                     - fluid_pp(i)%pi_inf) / fluid_pp(i)%gamma
                                    ! Physical pressure?
                                    IF (pres_K_init(i) .LE. -(1d0 - 1d-8)*pres_inf(i) + 1d-8) pres_K_init(i) = -(1d0 - 1d-8)*pres_inf(i) + 1d-8
                                ELSE
                                    pres_K_init(i) = 0d0
                                END IF
                                pres_relax = pres_relax + q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l)*pres_K_init(i)
                            END DO
! IF (j == 499 .OR. j == 500) print*, j, pres_relax, q_cons_vf(1+adv_idx%beg-1)%sf(j,k,l), q_cons_vf(2+adv_idx%beg-1)%sf(j,k,l), pres_K_init(1), pres_K_init(2), q_cons_vf(1+cont_idx%beg-1)%sf(j,k,l), q_cons_vf(2+cont_idx%beg-1)%sf(j,k,l) ! //KS//
                            ! Iterative process for relaxed pressure determination
                            iter    = 0
                            f_pres  = 1d-9
                            df_pres = 1d9
                            DO i = 1, num_fluids
                                rho_K_s(i) = 0d0
                            END DO

                            DO WHILE (DABS(f_pres) .GT. 1d-10)
                                pres_relax = pres_relax - f_pres / df_pres

                                ! Convergence?
                                iter = iter + 1
                                IF (iter == 50) THEN
                                    PRINT '(A)', 'Pressure relaxation procedure failed to converge to a solution. Exiting ...'
                                    CALL s_mpi_abort()
                                END IF

                                ! Physical pressure?
                                DO i = 1, num_fluids
                                    IF (pres_relax .LE. -(1d0 - 1d-8)*pres_inf(i) + 1d-8) pres_relax = -(1d0 - 1d-8)*pres_inf(i) + 1d0
                                END DO

                                ! Newton-Raphson method
                                f_pres  = -1d0
                                df_pres = 0d0
                                DO i = 1, num_fluids
                                    IF (q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l) .GT. sgm_eps) THEN
                                        rho_K_s(i) = q_cons_vf(i+cont_idx%beg-1)%sf(j,k,l) / MAX(q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l),sgm_eps) &
                                                      * ((pres_relax+pres_inf(i))/(pres_K_init(i)+pres_inf(i)))**(1d0/gamma_min(i))
                                        f_pres      = f_pres  + q_cons_vf(i+cont_idx%beg-1)%sf(j,k,l) / rho_K_s(i)
                                        df_pres     = df_pres - q_cons_vf(i+cont_idx%beg-1)%sf(j,k,l) / (gamma_min(i)*rho_K_s(i)*(pres_relax+pres_inf(i)))
                                    END IF
                                END DO

                            END DO
! IF (j == 827) print*, j, pres_relax, q_cons_vf(1+adv_idx%beg-1)%sf(j,k,l), q_cons_vf(2+adv_idx%beg-1)%sf(j,k,l), rho_K_s(1), rho_K_s(2) ! //KS//
                            ! Cell update of the volume fraction
                            DO i = 1, num_fluids
                                IF (q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l) .GT. sgm_eps) q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l) = q_cons_vf(i+cont_idx%beg-1)%sf(j,k,l) / rho_K_s(i)
                            END DO
! IF (j == 499 .OR. j == 500) print*, j, pres_relax, q_cons_vf(1+adv_idx%beg-1)%sf(j,k,l), q_cons_vf(2+adv_idx%beg-1)%sf(j,k,l), rho_K_s(1), rho_K_s(2) ! //KS//
                            ! Numerical correction of the volume fractions
                        END IF

                        ! ==================================================================
                        

                        ! Mixture-total-energy correction ==================================

                        ! The mixture-total-energy correction of the mixture pressure P is not necessary here
                        ! because the primitive variables are directly recovered later on by the conservative
                        ! variables (see s_convert_conservative_to_primitive_variables called in s_compute_rhs).
                        ! However, the internal-energy equations should be reset with the corresponding mixture
                        ! pressure from the correction. This step is carried out below.

                     
                        CALL s_convert_to_mixture_variables( q_cons_vf, rho, &
                                                             gamma, pi_inf,  &
                                                             Re, We, j,k,l   )
                     
                        dyn_pres = 0d0
                        DO i = mom_idx%beg, mom_idx%end
                            dyn_pres = dyn_pres + 5d-1*q_cons_vf(i)%sf(j,k,l)*q_cons_vf(i)%sf(j,k,l) / MAX(rho,sgm_eps)
                        END DO

                        E_We = 0d0
                        ! //KS// To do later on when capillary effects will be undertaken
                        ! IF (We_size > 0 .AND. (We_riemann_flux .OR. We_rhs_flux)) THEN
                        !      DO i = 1, We_size
                        !         E_We = E_We &
                        !                + q_cons_vf(E_idx+We_idx(i,1))%sf(j,k,l) * &
                        !                      gm_alpha_vf(We_idx(i,2))%sf(j,k,l) / &
                        !                            We(We_idx(i,1),We_idx(i,2))   &
                        !                + q_cons_vf(E_idx+We_idx(i,2))%sf(j,k,l) * &
                        !                      gm_alpha_vf(We_idx(i,1))%sf(j,k,l) / &
                        !                            We(We_idx(i,1),We_idx(i,2))
                        !      END DO
                        ! END IF
                     
                        pres_relax = (q_cons_vf(E_idx)%sf(j,k,l) - dyn_pres - pi_inf - E_We) / gamma

                        DO i = 1, num_fluids
                            q_cons_vf(i+internalEnergies_idx%beg-1)%sf(j,k,l) = q_cons_vf(i+adv_idx%beg-1)%sf(j,k,l) * (fluid_pp(i)%gamma*pres_relax + fluid_pp(i)%pi_inf)
                        END DO
                        ! ==================================================================
! IF (j == 499 .OR. j == 500) print*, j, pres_relax ! //KS//
                    END DO
                END DO
            END DO

        END SUBROUTINE s_pressure_relaxation_procedure ! -----------------------


        SUBROUTINE s_get_tvd_flux(q_cons_vf, q_prim_vf, rhs_vf,i) ! -------
            
            ! Cell-average conservative variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: q_cons_vf
            
            ! Cell-average primitive variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: q_prim_vf
            
            ! Cell-average RHS variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: rhs_vf

            INTEGER, INTENT(in) :: i
            ! Generic loop iterators
            INTEGER :: j,k,l,r

            REAL(kind(0d0)) :: top, bottom
            
            DO l = iz%beg, iz%end
                DO k = iy%beg, iy%end
                    DO j = ix%beg, ix%end
                        ! Upwind direction for slope chosen based on contact velocity
                        ! from Riemann solver (stored in adv_idx of flux_src). Slope
                        ! is calculated using the first volume fraction.
                        IF (hi_flux_src_ndqp(i,0,0)%vf(adv_idx%beg)%sf(j,k,l) >= 0d0) THEN
                               top = q_prim_qp(0,0,0)%vf(adv_idx%beg)%sf( j ,k,l) - &
                                 q_prim_qp(0,0,0)%vf(adv_idx%beg)%sf(j-1,k,l)
                            bottom = q_prim_qp(0,0,0)%vf(adv_idx%beg)%sf(j+1,k,l) - &
                                 q_prim_qp(0,0,0)%vf(adv_idx%beg)%sf( j ,k,l)
                        ELSE
                               top = q_prim_qp(0,0,0)%vf(adv_idx%beg)%sf(j+2,k,l) - &
                                 q_prim_qp(0,0,0)%vf(adv_idx%beg)%sf(j+1,k,l)
                            bottom = q_prim_qp(0,0,0)%vf(adv_idx%beg)%sf(j+1,k,l) - &
                                 q_prim_qp(0,0,0)%vf(adv_idx%beg)%sf( j ,k,l)
                        END IF
            
                        CALL s_compute_flux_lim(top,bottom,flux_lim_func(j,k,l))
                    END DO
                END DO
            END DO

            ! Compile final flux and flux source
            DO k = ipsi%beg, ipsi%end, 2
                DO j = ichi%beg, ichi%end, 2
                    DO l = 1, sys_size
                            flux_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) = &
                         lo_flux_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) + &
                                        flux_lim_func(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) * &
                        (hi_flux_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) - &
                         lo_flux_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
                    END DO

                    IF(ANY(Re_size > 0) .OR. We_size > 0) THEN
                        DO l = mom_idx%beg, E_idx
                            flux_src_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) = &
                         lo_flux_src_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) + &
                                            flux_lim_func(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) * &
                        (hi_flux_src_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) - &
                         lo_flux_src_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
                        END DO
                    END IF
                    DO l = adv_idx%beg, adv_idx%end
                            flux_src_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) = &
                         lo_flux_src_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) + &
                                            flux_lim_func(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) * &
                        (hi_flux_src_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) - &
                         lo_flux_src_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
                    END DO
                    IF (cyl_coord) THEN
                        DO l = 1, sys_size
                            flux_gsrc_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) = &
                         lo_flux_gsrc_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) + &
                                             flux_lim_func(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) * &
                        (hi_flux_gsrc_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end) - &
                         lo_flux_gsrc_ndqp(i,j,k)%vf(l)%sf(ix%beg:ix%end,iy%beg:iy%end,iz%beg:iz%end))
                        END DO
                    END IF
                END DO
            END DO

            iv%beg = 1; iv%end = adv_idx%end
        
            CALL s_average_cell_boundary_values(flux_ndqp(i,:,:))
            
            IF(ANY(Re_size > 0) .OR. We_size > 0) THEN
               iv%beg = mom_idx%beg
            ELSE
               iv%beg = adv_idx%beg
            END IF
            
            IF(riemann_solver /= 1) iv%end = adv_idx%beg
            
            CALL s_average_cell_boundary_values(flux_src_ndqp(i,:,:))
            CALL s_average_cell_boundary_values(flux_gsrc_ndqp(i,:,:))
        END SUBROUTINE s_get_tvd_flux

        SUBROUTINE s_get_viscous(q_cons_vf, q_prim_vf, rhs_vf) ! -------
            
            ! Cell-average conservative variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: q_cons_vf
            
            ! Cell-average primitive variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: q_prim_vf
            
            ! Cell-average RHS variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: rhs_vf

            ! Generic loop iterators
            INTEGER :: i,j,k,l,r

                DO i = 1, num_dims
                    ! WENO reconstruct variables to cell boundaries
                    IF(weno_vars == 1) THEN
                     
                        iv%beg = 1; iv%end = mom_idx%end
                        
                        CALL s_reconstruct_cell_boundary_values(       &
                                   q_cons_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                  qL_cons_ndqp(i,:,:), &
                                                  qR_cons_ndqp(i,:,:), &
                                                          dflt_int, i  )
                        
                        DO k = ipsi%beg, ipsi%end, 2
                            DO j = ichi%beg, ichi%end, 2
                                DO l = mom_idx%beg, mom_idx%end
                                 
                                    qL_prim_ndqp(i,j,k)%vf(l)%sf = sgm_eps
                                    qR_prim_ndqp(i,j,k)%vf(l)%sf = sgm_eps
                                    
                                    DO r = 1, cont_idx%end
                                        qL_prim_ndqp(i,j,k)%vf(l)%sf = &
                                        qL_prim_ndqp(i,j,k)%vf(l)%sf + &
                                        qL_cons_ndqp(i,j,k)%vf(r)%sf
                                        qR_prim_ndqp(i,j,k)%vf(l)%sf = &
                                        qR_prim_ndqp(i,j,k)%vf(l)%sf + &
                                        qR_cons_ndqp(i,j,k)%vf(r)%sf
                                    END DO
                                    
                                    qL_prim_ndqp(i,j,k)%vf(l)%sf = &
                                    qL_cons_ndqp(i,j,k)%vf(l)%sf / &
                                    qL_prim_ndqp(i,j,k)%vf(l)%sf
                                    qR_prim_ndqp(i,j,k)%vf(l)%sf = &
                                    qR_cons_ndqp(i,j,k)%vf(l)%sf / &
                                    qR_prim_ndqp(i,j,k)%vf(l)%sf
                                 
                                END DO
                            END DO
                        END DO
                     
                    ELSE
                     
                        iv%beg = mom_idx%beg; iv%end = mom_idx%end
                     
                        CALL s_reconstruct_cell_boundary_values(       &
                                   q_prim_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                  qL_prim_ndqp(i,:,:), &
                                                  qR_prim_ndqp(i,:,:), &
                                                          dflt_int, i  )
                     
                    END IF
                  
                    iv%beg = mom_idx%beg; iv%end = mom_idx%end
                  
                    CALL s_average_cell_boundary_values(qL_prim_ndqp(i,:,:))
                    CALL s_average_cell_boundary_values(qR_prim_ndqp(i,:,:))
                END DO

                IF (weno_Re_flux) THEN
                ! Compute velocity gradient at cell centers using scalar
                ! divergence theorem
                    DO i = 1, num_dims
                      
                        IF(i == 1) THEN
                            CALL s_apply_scalar_divergence_theorem(           &
                                     qL_prim_ndqp(i,0,0)%vf(iv%beg:iv%end),   &
                                     qR_prim_ndqp(i,0,0)%vf(iv%beg:iv%end),   &
                                    dq_prim_dx_qp(0,0,0)%vf(iv%beg:iv%end), i )
                        ELSEIF(i == 2) THEN
                            CALL s_apply_scalar_divergence_theorem(           &
                                     qL_prim_ndqp(i,0,0)%vf(iv%beg:iv%end),   &
                                     qR_prim_ndqp(i,0,0)%vf(iv%beg:iv%end),   &
                                    dq_prim_dy_qp(0,0,0)%vf(iv%beg:iv%end), i )
                        ELSE
                            CALL s_apply_scalar_divergence_theorem(           &
                                     qL_prim_ndqp(i,0,0)%vf(iv%beg:iv%end),   &
                                     qR_prim_ndqp(i,0,0)%vf(iv%beg:iv%end),   &
                                    dq_prim_dz_qp(0,0,0)%vf(iv%beg:iv%end), i )
                        END IF

                    END DO
                      
                ELSE ! Compute velocity gradient at cell centers using finite differences

                    iv%beg = mom_idx%beg; iv%end = mom_idx%end

                    DO k = iv%beg, iv%end

                        DO j = ix%beg+1, ix%end
                            dqL_prim_dx_ndqp(1,0,0)%vf(k)%sf( j ,:,:) = &
                                  (q_prim_qp(0,0,0)%vf(k)%sf( j ,:,:) - &
                                   q_prim_qp(0,0,0)%vf(k)%sf(j-1,:,:))/ &
                                                    (x_cc(j) - x_cc(j-1))
                        END DO
                        DO j = ix%beg, ix%end-1
                            dqR_prim_dx_ndqp(1,0,0)%vf(k)%sf( j ,:,:) = &
                                  (q_prim_qp(0,0,0)%vf(k)%sf(j+1,:,:) - &
                                   q_prim_qp(0,0,0)%vf(k)%sf( j ,:,:))/ &
                                                    (x_cc(j+1) - x_cc(j))
                        END DO

                        IF (n > 0) THEN

                            DO j = iy%beg+1, iy%end
                                dqL_prim_dy_ndqp(2,0,0)%vf(k)%sf(:, j ,:) = &
                                      (q_prim_qp(0,0,0)%vf(k)%sf(:, j ,:) - &
                                       q_prim_qp(0,0,0)%vf(k)%sf(:,j-1,:))/ &
                                                        (y_cc(j) - y_cc(j-1))
                            END DO
                            DO j = iy%beg, iy%end-1
                                dqR_prim_dy_ndqp(2,0,0)%vf(k)%sf(:, j ,:) = &
                                      (q_prim_qp(0,0,0)%vf(k)%sf(:,j+1,:) - &
                                       q_prim_qp(0,0,0)%vf(k)%sf(:, j ,:))/ &
                                                        (y_cc(j+1) - y_cc(j))
                            END DO
                            DO j = iy%beg+1, iy%end
                                dqL_prim_dx_ndqp(2,0,0)%vf(k)%sf(ix%beg+1:ix%end-1, j ,:) = &
                               (dqL_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1, j ,:) + &
                                dqR_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1, j ,:) + &
                                dqL_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,j-1,:) + &
                                dqR_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,j-1,:))
                            END DO
                            DO j = iy%beg, iy%end-1
                                dqR_prim_dx_ndqp(2,0,0)%vf(k)%sf(ix%beg+1:ix%end-1, j ,:) = &
                               (dqL_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,j+1,:) + &
                                dqR_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,j+1,:) + &
                                dqL_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1, j ,:) + &
                                dqR_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1, j ,:))
                            END DO
                            DO j = ix%beg+1, ix%end
                                dqL_prim_dy_ndqp(1,0,0)%vf(k)%sf( j ,iy%beg+1:iy%end-1,:) = &
                               (dqL_prim_dy_ndqp(2,0,0)%vf(k)%sf( j ,iy%beg+1:iy%end-1,:) + &
                                dqR_prim_dy_ndqp(2,0,0)%vf(k)%sf( j ,iy%beg+1:iy%end-1,:) + &
                                dqL_prim_dy_ndqp(2,0,0)%vf(k)%sf(j-1,iy%beg+1:iy%end-1,:) + &
                                dqR_prim_dy_ndqp(2,0,0)%vf(k)%sf(j-1,iy%beg+1:iy%end-1,:))
                            END DO
                            DO j = ix%beg, ix%end-1
                                dqR_prim_dy_ndqp(1,0,0)%vf(k)%sf( j ,iy%beg+1:iy%end-1,:) = &
                               (dqL_prim_dy_ndqp(2,0,0)%vf(k)%sf(j+1,iy%beg+1:iy%end-1,:) + &
                                dqR_prim_dy_ndqp(2,0,0)%vf(k)%sf(j+1,iy%beg+1:iy%end-1,:) + &
                                dqL_prim_dy_ndqp(2,0,0)%vf(k)%sf( j ,iy%beg+1:iy%end-1,:) + &
                                dqR_prim_dy_ndqp(2,0,0)%vf(k)%sf( j ,iy%beg+1:iy%end-1,:))
                            END DO
                            dqL_prim_dx_ndqp(2,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,iy%beg+1:iy%end,:) = 25d-2 * &
                            dqL_prim_dx_ndqp(2,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,iy%beg+1:iy%end,:)
                            dqR_prim_dx_ndqp(2,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,iy%beg:iy%end-1,:) = 25d-2 * &
                            dqR_prim_dx_ndqp(2,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,iy%beg:iy%end-1,:)
                            dqL_prim_dy_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end,iy%beg+1:iy%end-1,:) = 25d-2 * &
                            dqL_prim_dy_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end,iy%beg+1:iy%end-1,:)
                            dqR_prim_dy_ndqp(1,0,0)%vf(k)%sf(ix%beg:ix%end-1,iy%beg+1:iy%end-1,:) = 25d-2 * &
                            dqR_prim_dy_ndqp(1,0,0)%vf(k)%sf(ix%beg:ix%end-1,iy%beg+1:iy%end-1,:)

                            IF (p > 0) THEN

                                DO j = iz%beg+1, iz%end
                                    dqL_prim_dz_ndqp(3,0,0)%vf(k)%sf(:,:, j ) = &
                                          (q_prim_qp(0,0,0)%vf(k)%sf(:,:, j ) - &
                                           q_prim_qp(0,0,0)%vf(k)%sf(:,:,j-1))/ &
                                                            (z_cc(j) - z_cc(j-1))
                                END DO
                                DO j = iz%beg, iz%end-1
                                    dqR_prim_dz_ndqp(3,0,0)%vf(k)%sf(:,:, j ) = &
                                          (q_prim_qp(0,0,0)%vf(k)%sf(:,:,j+1) - &
                                           q_prim_qp(0,0,0)%vf(k)%sf(:,:, j ))/ &
                                                            (z_cc(j+1) - z_cc(j))
                                END DO
                                DO j = ix%beg+1, ix%end
                                    dqL_prim_dz_ndqp(1,0,0)%vf(k)%sf( j ,:,iz%beg+1:iz%end-1) = &
                                   (dqL_prim_dz_ndqp(3,0,0)%vf(k)%sf( j ,:,iz%beg+1:iz%end-1) + &
                                    dqR_prim_dz_ndqp(3,0,0)%vf(k)%sf( j ,:,iz%beg+1:iz%end-1) + &
                                    dqL_prim_dz_ndqp(3,0,0)%vf(k)%sf(j-1,:,iz%beg+1:iz%end-1) + &
                                    dqR_prim_dz_ndqp(3,0,0)%vf(k)%sf(j-1,:,iz%beg+1:iz%end-1))
                                END DO
                                DO j = ix%beg, ix%end-1
                                    dqR_prim_dz_ndqp(1,0,0)%vf(k)%sf( j ,:,iz%beg+1:iz%end-1) = &
                                   (dqL_prim_dz_ndqp(3,0,0)%vf(k)%sf(j+1,:,iz%beg+1:iz%end-1) + &
                                    dqR_prim_dz_ndqp(3,0,0)%vf(k)%sf(j+1,:,iz%beg+1:iz%end-1) + &
                                    dqL_prim_dz_ndqp(3,0,0)%vf(k)%sf( j ,:,iz%beg+1:iz%end-1) + &
                                    dqR_prim_dz_ndqp(3,0,0)%vf(k)%sf( j ,:,iz%beg+1:iz%end-1))
                                END DO
                                DO j = iy%beg+1, iy%end
                                    dqL_prim_dz_ndqp(2,0,0)%vf(k)%sf(:, j ,iz%beg+1:iz%end-1) = &
                                   (dqL_prim_dz_ndqp(3,0,0)%vf(k)%sf(:, j ,iz%beg+1:iz%end-1) + &
                                    dqR_prim_dz_ndqp(3,0,0)%vf(k)%sf(:, j ,iz%beg+1:iz%end-1) + &
                                    dqL_prim_dz_ndqp(3,0,0)%vf(k)%sf(:,j-1,iz%beg+1:iz%end-1) + &
                                    dqR_prim_dz_ndqp(3,0,0)%vf(k)%sf(:,j-1,iz%beg+1:iz%end-1))
                                END DO
                                DO j = iy%beg, iy%end-1
                                    dqR_prim_dz_ndqp(2,0,0)%vf(k)%sf(:, j ,iz%beg+1:iz%end-1) = &
                                   (dqL_prim_dz_ndqp(3,0,0)%vf(k)%sf(:,j+1,iz%beg+1:iz%end-1) + &
                                    dqR_prim_dz_ndqp(3,0,0)%vf(k)%sf(:,j+1,iz%beg+1:iz%end-1) + &
                                    dqL_prim_dz_ndqp(3,0,0)%vf(k)%sf(:, j ,iz%beg+1:iz%end-1) + &
                                    dqR_prim_dz_ndqp(3,0,0)%vf(k)%sf(:, j ,iz%beg+1:iz%end-1))
                                END DO
                                DO j = iz%beg+1, iz%end
                                    dqL_prim_dy_ndqp(3,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1, j ) = &
                                   (dqL_prim_dy_ndqp(2,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1, j ) + &
                                    dqR_prim_dy_ndqp(2,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1, j ) + &
                                    dqL_prim_dy_ndqp(2,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1,j-1) + &
                                    dqR_prim_dy_ndqp(2,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1,j-1))
                                END DO
                                DO j = iz%beg, iz%end-1
                                    dqR_prim_dy_ndqp(3,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1, j ) = &
                                   (dqL_prim_dy_ndqp(2,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1,j+1) + &
                                    dqR_prim_dy_ndqp(2,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1,j+1) + &
                                    dqL_prim_dy_ndqp(2,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1, j ) + &
                                    dqR_prim_dy_ndqp(2,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1, j ))
                                END DO
                                DO j = iz%beg+1, iz%end
                                    dqL_prim_dx_ndqp(3,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:, j ) = &
                                   (dqL_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:, j ) + &
                                    dqR_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:, j ) + &
                                    dqL_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:,j-1) + &
                                    dqR_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:,j-1))
                                END DO
                                DO j = iz%beg, iz%end-1
                                    dqR_prim_dx_ndqp(3,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:, j ) = &
                                   (dqL_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:,j+1) + &
                                    dqR_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:,j+1) + &
                                    dqL_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:, j ) + &
                                    dqR_prim_dx_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:, j ))
                                END DO

                                dqL_prim_dz_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end,:,iz%beg+1:iz%end-1) = 25d-2 * &
                                dqL_prim_dz_ndqp(1,0,0)%vf(k)%sf(ix%beg+1:ix%end,:,iz%beg+1:iz%end-1)
                                dqR_prim_dz_ndqp(1,0,0)%vf(k)%sf(ix%beg:ix%end-1,:,iz%beg+1:iz%end-1) = 25d-2 * &
                                dqR_prim_dz_ndqp(1,0,0)%vf(k)%sf(ix%beg:ix%end-1,:,iz%beg+1:iz%end-1)
                                dqL_prim_dz_ndqp(2,0,0)%vf(k)%sf(:,iy%beg+1:iy%end,iz%beg+1:iz%end-1) = 25d-2 * &
                                dqL_prim_dz_ndqp(2,0,0)%vf(k)%sf(:,iy%beg+1:iy%end,iz%beg+1:iz%end-1)
                                dqR_prim_dz_ndqp(2,0,0)%vf(k)%sf(:,iy%beg:iy%end-1,iz%beg+1:iz%end-1) = 25d-2 * &
                                dqR_prim_dz_ndqp(2,0,0)%vf(k)%sf(:,iy%beg:iy%end-1,iz%beg+1:iz%end-1)
                                dqL_prim_dy_ndqp(3,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1,iz%beg+1:iz%end) = 25d-2 * &
                                dqL_prim_dy_ndqp(3,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1,iz%beg+1:iz%end)
                                dqR_prim_dy_ndqp(3,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1,iz%beg:iz%end-1) = 25d-2 * &
                                dqR_prim_dy_ndqp(3,0,0)%vf(k)%sf(:,iy%beg+1:iy%end-1,iz%beg:iz%end-1)
                                dqL_prim_dx_ndqp(3,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:,iz%beg+1:iz%end) = 25d-2 * &
                                dqL_prim_dx_ndqp(3,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:,iz%beg+1:iz%end)
                                dqR_prim_dx_ndqp(3,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:,iz%beg:iz%end-1) = 25d-2 * &
                                dqR_prim_dx_ndqp(3,0,0)%vf(k)%sf(ix%beg+1:ix%end-1,:,iz%beg:iz%end-1)

                                CALL s_compute_fd_gradient(    q_prim_qp(0,0,0)%vf(k), &
                                                           dq_prim_dx_qp(0,0,0)%vf(k), &
                                                           dq_prim_dy_qp(0,0,0)%vf(k), &
                                                           dq_prim_dz_qp(0,0,0)%vf(k), &
                                                               gm_vel_qp(0,0,0)%vf(k)  )

                            ELSE

                                CALL s_compute_fd_gradient(    q_prim_qp(0,0,0)%vf(k), &
                                                           dq_prim_dx_qp(0,0,0)%vf(k), &
                                                           dq_prim_dy_qp(0,0,0)%vf(k), &
                                                           dq_prim_dy_qp(0,0,0)%vf(k), &
                                                               gm_vel_qp(0,0,0)%vf(k)  )

                            END IF

                        ELSE

                            CALL s_compute_fd_gradient(    q_prim_qp(0,0,0)%vf(k), &
                                                       dq_prim_dx_qp(0,0,0)%vf(k), &
                                                       dq_prim_dx_qp(0,0,0)%vf(k), &
                                                       dq_prim_dx_qp(0,0,0)%vf(k), &
                                                           gm_vel_qp(0,0,0)%vf(k)  )

                        END IF

                    END DO

                END IF

        END SUBROUTINE s_get_viscous


        SUBROUTINE s_get_crv(q_cons_vf, q_prim_vf, rhs_vf) ! -------
            
            ! Cell-average conservative variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: q_cons_vf
            
            ! Cell-average primitive variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: q_prim_vf
            
            ! Cell-average RHS variables
            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: rhs_vf

            ! Generic loop iterators
            INTEGER :: i,j,k,l,r

            DO l = 1, crv_size
               
                iv%beg = E_idx + crv_idx(l); iv%end = iv%beg
               
            ! ==================================================================
                DO i = 1, num_dims

                    ! Reconstruct the volume fraction at the cell boundaries
                    CALL s_reconstruct_cell_boundary_values(         &
                                 q_prim_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                qL_prim_ndqp(i,:,:), &
                                                qR_prim_ndqp(i,:,:), &
                                                        dflt_int, i  )
                    ! Average the quadrature points at the cell boundaries to
                    ! obtain higher-order estimate of the volume fraction
                    CALL s_average_cell_boundary_values(qL_prim_ndqp(i,:,:))
                    CALL s_average_cell_boundary_values(qR_prim_ndqp(i,:,:))
                  
                END DO
            ! ==================================================================

                IF (lsq_deriv) THEN
                    CALL s_reconstruct_cell_interior_values(dq_prim_dx_qp)
                    IF (n > 0) THEN
                        CALL s_reconstruct_cell_interior_values(dq_prim_dy_qp)
                        IF (p > 0) THEN
                            CALL s_reconstruct_cell_interior_values(dq_prim_dz_qp)
                        END IF
                    END IF
                ELSE
                    ! Apply the scalar divergence theorem to compute the volume
                    ! fraction gradient at the cell center
                    CALL s_apply_scalar_divergence_theorem( &
                            qL_prim_ndqp(1,0,0)%vf(iv%beg:iv%end),   &
                            qR_prim_ndqp(1,0,0)%vf(iv%beg:iv%end),   &
                            dq_prim_dx_qp(0,0,0)%vf(iv%beg:iv%end), 1 )
            
                    CALL s_reconstruct_cell_interior_values(dq_prim_dx_qp)

                    IF (n > 0) THEN
                        CALL s_apply_scalar_divergence_theorem( &
                                qL_prim_ndqp(2,0,0)%vf(iv%beg:iv%end),   &
                                qR_prim_ndqp(2,0,0)%vf(iv%beg:iv%end),   &
                                dq_prim_dy_qp(0,0,0)%vf(iv%beg:iv%end), 2 )
            
                        CALL s_reconstruct_cell_interior_values(dq_prim_dy_qp)

                        IF (p > 0) THEN
                            CALL s_apply_scalar_divergence_theorem( &
                                    qL_prim_ndqp(3,0,0)%vf(iv%beg:iv%end),   &
                                    qR_prim_ndqp(3,0,0)%vf(iv%beg:iv%end),   &
                                    dq_prim_dz_qp(0,0,0)%vf(iv%beg:iv%end), 3 )
            
                            CALL s_reconstruct_cell_interior_values(dq_prim_dz_qp)
                        END IF
                    END IF
                END IF

            ! ==================================================================
               
                ! Compute the gradient magnitude at the cell center
                DO k = itau%beg, itau%end, 2
                     DO j = iksi%beg, iksi%end, 2
                         DO i = ieta%beg, ieta%end, 2
                             CALL s_compute_gradient_magnitude(              &
                                     dq_prim_dx_qp(i,j,k)%vf(iv%beg:iv%end), &
                                     dq_prim_dy_qp(i,j,k)%vf(iv%beg:iv%end), &
                                     dq_prim_dz_qp(i,j,k)%vf(iv%beg:iv%end), &
                                       gm_alpha_qp(i,j,k)%vf( crv_idx(l):    &
                                                              crv_idx(l)  )  )
                         END DO
                     END DO
                END DO
            
            ! ==================================================================
               
                ! Reconstruct the volume fraction gradient at the cell
                ! boundaries
                DO i = 1, num_dims
                   
                     CALL s_reconstruct_cell_boundary_values(           &
                                dq_prim_dx_qp(0,0,0)%vf(iv%beg:iv%end), &
                                               dqL_prim_dx_ndqp(i,:,:), &
                                               dqR_prim_dx_ndqp(i,:,:), &
                                                           dflt_int, i  )
                     IF (n > 0) THEN
                         CALL s_reconstruct_cell_boundary_values(           &
                                    dq_prim_dy_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                   dqL_prim_dy_ndqp(i,:,:), &
                                                   dqR_prim_dy_ndqp(i,:,:), &
                                                               dflt_int, i  )
                         IF (p > 0) THEN
                             CALL s_reconstruct_cell_boundary_values(         &
                                      dq_prim_dz_qp(0,0,0)%vf(iv%beg:iv%end), &
                                                     dqL_prim_dz_ndqp(i,:,:), &
                                                     dqR_prim_dz_ndqp(i,:,:), &
                                                                 dflt_int, i  )
                         END IF
                     END IF

                END DO
            ! ==================================================================

                ! Compute the gradient magnitude at the cell boundaries
                DO k = ipsi%beg, ipsi%end, 2
                    DO j = ichi%beg, ichi%end, 2
                        DO i = 1, num_dims
                            CALL s_compute_gradient_magnitude(               &
                                  dqL_prim_dx_ndqp(i,j,k)%vf(iv%beg:iv%end), &
                                  dqL_prim_dy_ndqp(i,j,k)%vf(iv%beg:iv%end), &
                                  dqL_prim_dz_ndqp(i,j,k)%vf(iv%beg:iv%end), &
                                    gm_alphaL_ndqp(i,j,k)%vf( crv_idx(l):    &
                                                              crv_idx(l)  )  )
                            CALL s_compute_gradient_magnitude(               &
                                  dqR_prim_dx_ndqp(i,j,k)%vf(iv%beg:iv%end), &
                                  dqR_prim_dy_ndqp(i,j,k)%vf(iv%beg:iv%end), &
                                  dqR_prim_dz_ndqp(i,j,k)%vf(iv%beg:iv%end), &
                                    gm_alphaR_ndqp(i,j,k)%vf( crv_idx(l):    &
                                                              crv_idx(l)  )  )
                        END DO
                    END DO
                END DO
            ! ==================================================================
                IF (alt_crv) THEN
            ! ==================================================================

                    ! Compute the normalized volume fraction gradient at the
                    ! cell center
                    DO k = itau%beg, itau%end, 2
                        DO j = iksi%beg, iksi%end, 2
                            DO i = ieta%beg, ieta%end, 2
                                un_alpha_x_qp(i,j,k)%vf(crv_idx(l))%sf = &
                                    dq_prim_dx_qp(i,j,k)%vf(E_idx+crv_idx(l))%sf / &
                                    MAX(gm_alpha_qp(i,j,k)%vf(crv_idx(l))%sf,sgm_eps)
                                IF (n > 0) THEN
                                    un_alpha_y_qp(i,j,k)%vf(crv_idx(l))%sf = &
                                        dq_prim_dy_qp(i,j,k)%vf(E_idx+crv_idx(l))%sf / &
                                        MAX(gm_alpha_qp(i,j,k)%vf(crv_idx(l))%sf,sgm_eps)
                                    IF (p > 0) THEN
                                        un_alpha_z_qp(i,j,k)%vf(crv_idx(l))%sf = &
                                            dq_prim_dz_qp(i,j,k)%vf(E_idx+crv_idx(l))%sf / &
                                            MAX(gm_alpha_qp(i,j,k)%vf(crv_idx(l))%sf,sgm_eps)
                                    END IF
                                END IF
                            END DO
                        END DO
                    END DO
            ! ==================================================================

                    ! Apply the scalar divergence theorem to compute the gradient
                    ! of the gradient magnitude at the cell center
                    CALL s_apply_scalar_divergence_theorem(             &
                            gm_alphaL_ndqp(1,0,0)%vf(crv_idx(l):crv_idx(l)),&
                            gm_alphaR_ndqp(1,0,0)%vf(crv_idx(l):crv_idx(l)),&
                            dgm_alpha_dx_qp(0,0,0)%vf(crv_idx(l):crv_idx(l)),1)
                    ! CALL s_reconstruct_cell_interior_values(dgm_alpha_dx_qp)
        
                    IF (n > 0) THEN
                        CALL s_apply_scalar_divergence_theorem(             &
                                gm_alphaL_ndqp(2,0,0)%vf(crv_idx(l):crv_idx(l)),&
                                gm_alphaR_ndqp(2,0,0)%vf(crv_idx(l):crv_idx(l)),&
                                dgm_alpha_dy_qp(0,0,0)%vf(crv_idx(l):crv_idx(l)),2)
                        ! CALL s_reconstruct_cell_interior_values(dgm_alpha_dy_qp)
                        IF (p > 0) THEN
                            CALL s_apply_scalar_divergence_theorem(             &
                                    gm_alphaL_ndqp(3,0,0)%vf(crv_idx(l):crv_idx(l)),&
                                    gm_alphaR_ndqp(3,0,0)%vf(crv_idx(l):crv_idx(l)),&
                                    dgm_alpha_dz_qp(0,0,0)%vf(crv_idx(l):crv_idx(l)),3)
                            ! CALL s_reconstruct_cell_interior_values(dgm_alpha_dz_qp)
                        END IF
                    END IF
            ! ==================================================================
                    
                    !Apply the vector divergence theorem to compute the  
                    ! Laplacian of the volume fraction at the cell center
                    CALL s_apply_vector_divergence_theorem(     &
                                        dqL_prim_dx_ndqp,       &
                                        dqL_prim_dy_ndqp,       &
                                        dqL_prim_dz_ndqp,       &
                                        dqR_prim_dx_ndqp,       &
                                        dqR_prim_dy_ndqp,       &
                                        dqR_prim_dz_ndqp,       &
                                        laplacian_vf(crv_idx(l):crv_idx(l)))
            ! ==================================================================
              
                    ! Compute alternate volume fraction curvature
                    alt_kappa_vf(crv_idx(l))%sf = ( un_alpha_x_qp(0,0,0)%vf(crv_idx(l))%sf * &
                                    dgm_alpha_dx_qp(0,0,0)%vf(crv_idx(l))%sf + &
                                    un_alpha_y_qp(0,0,0)%vf(crv_idx(l))%sf * &
                                    dgm_alpha_dy_qp(0,0,0)%vf(crv_idx(l))%sf + &
                                    un_alpha_z_qp(0,0,0)%vf(crv_idx(l))%sf * &
                                    dgm_alpha_dz_qp(0,0,0)%vf(crv_idx(l))%sf - &
                                    laplacian_vf(crv_idx(l))%sf ) / &
                                    MAX(gm_alpha_qp(0,0,0)%vf(crv_idx(l))%sf,sgm_eps)
            ! ==================================================================
                ELSE
            ! ==================================================================
        
                    ! Compute the normalized volume fraction gradient at the cell
                    ! boundaries
                    DO k = ipsi%beg, ipsi%end, 2
                        DO j = ichi%beg, ichi%end, 2
                            un_alphaL_x_ndqp(1,j,k)%vf(crv_idx(l))%sf =           &
                                dqL_prim_dx_ndqp(1,j,k)%vf(E_idx+crv_idx(l))%sf / &
                                MAX(gm_alphaL_ndqp(1,j,k)%vf(crv_idx(l))%sf,sgm_eps)
                            un_alphaR_x_ndqp(1,j,k)%vf(crv_idx(l))%sf =           &
                                dqR_prim_dx_ndqp(1,j,k)%vf(E_idx+crv_idx(l))%sf / &
                                MAX(gm_alphaR_ndqp(1,j,k)%vf(crv_idx(l))%sf,sgm_eps)
                            IF (n > 0) THEN
                                un_alphaL_y_ndqp(2,j,k)%vf(crv_idx(l))%sf =           &
                                    dqL_prim_dy_ndqp(2,j,k)%vf(E_idx+crv_idx(l))%sf / &
                                    MAX(gm_alphaL_ndqp(2,j,k)%vf(crv_idx(l))%sf,sgm_eps)
                                un_alphaR_y_ndqp(2,j,k)%vf(crv_idx(l))%sf =           &
                                    dqR_prim_dy_ndqp(2,j,k)%vf(E_idx+crv_idx(l))%sf / &
                                    MAX(gm_alphaR_ndqp(2,j,k)%vf(crv_idx(l))%sf,sgm_eps)
                                IF (p > 0) THEN
                                    un_alphaL_z_ndqp(3,j,k)%vf(crv_idx(l))%sf =           &
                                        dqL_prim_dz_ndqp(3,j,k)%vf(E_idx+crv_idx(l))%sf / &
                                        MAX(gm_alphaL_ndqp(3,j,k)%vf(crv_idx(l))%sf,sgm_eps)
                                    un_alphaR_z_ndqp(3,j,k)%vf(crv_idx(l))%sf =           &
                                        dqR_prim_dz_ndqp(3,j,k)%vf(E_idx+crv_idx(l))%sf / &
                                        MAX(gm_alphaR_ndqp(3,j,k)%vf(crv_idx(l))%sf,sgm_eps)
                                END IF
                            END IF
                        END DO
                    END DO
            ! ==================================================================
               
                    iv%beg = crv_idx(l); iv%end = iv%beg
                    ! Average the quadrature points at the cell boundaries to
                    ! obtain higher-order estimate of the normalized volume fraction
                    ! gradient
                    CALL s_average_cell_boundary_values(un_alphaL_x_ndqp(1,:,:))
                    CALL s_average_cell_boundary_values(un_alphaR_x_ndqp(1,:,:))
                    IF (n > 0) THEN
                        CALL s_average_cell_boundary_values(un_alphaL_y_ndqp(2,:,:))
                        CALL s_average_cell_boundary_values(un_alphaR_y_ndqp(2,:,:))
                        IF(p > 0) THEN
                            CALL s_average_cell_boundary_values(un_alphaL_z_ndqp(3,:,:))
                            CALL s_average_cell_boundary_values(un_alphaR_z_ndqp(3,:,:))
                        END IF
                    END IF
            ! ==================================================================
                    IF (lsq_deriv .NEQV. .TRUE.) THEN 
                       ! Apply the vector divergence theorem to compute the divergence
                       ! of the normalized volume fraction gradient
                       CALL s_apply_vector_divergence_theorem(     &
                                                 un_alphaL_x_ndqp, &
                                                 un_alphaL_y_ndqp, &
                                                 un_alphaL_z_ndqp, &
                                                 un_alphaR_x_ndqp, &
                                                 un_alphaR_y_ndqp, &
                                                 un_alphaR_z_ndqp, &
                                           kappa_vf(iv%beg:iv%end) )
                       
                       kappa_vf(crv_idx(l))%sf = -kappa_vf(crv_idx(l))%sf
                    END IF
            ! ==================================================================
                END IF
            END DO
            ! END: Computing Volume Fraction Gradients and Curvatures ==========


        END SUBROUTINE s_get_crv



        SUBROUTINE s_populate_variables_buffers(v_vf) ! ---------------
        ! Description: The purpose of this procedure is to populate the buffers
        !              of the conservative variables, depending on the selected
        !              boundary conditions.

            TYPE(scalar_field), DIMENSION(sys_size), INTENT(INOUT) :: v_vf
            
            ! Generic loop iterators
            INTEGER :: i,j,k
            
            
            ! Population of Buffers in x-direction =============================
            
            IF(bc_x%beg <= -3) THEN         ! Ghost-cell extrap. BC at beginning
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        v_vf(i)%sf(-j,0:n,0:p) = &
                        v_vf(i)%sf( 0,0:n,0:p)
                    END DO
                END DO
                
            ELSEIF(bc_x%beg == -2) THEN     ! Symmetry BC at beginning
                
                DO j = 1, buff_size
                    
                    DO i = 1, cont_idx%end
                        v_vf(i)%sf(-j ,0:n,0:p) = &
                        v_vf(i)%sf(j-1,0:n,0:p)
                    END DO
                    
                    v_vf(mom_idx%beg)%sf(-j ,0:n,0:p) = &
                   -v_vf(mom_idx%beg)%sf(j-1,0:n,0:p)
                    
                    DO i = mom_idx%beg+1, sys_size
                        v_vf(i)%sf(-j ,0:n,0:p) = &
                        v_vf(i)%sf(j-1,0:n,0:p)
                    END DO
                    
                END DO
                
            ELSEIF(bc_x%beg == -1) THEN     ! Periodic BC at beginning
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        v_vf(i)%sf(  -j   ,0:n,0:p) = &
                        v_vf(i)%sf(m-(j-1),0:n,0:p)
                    END DO
                END DO
                
            ELSE                            ! Processor BC at beginning
                
                CALL s_mpi_sendrecv_conservative_variables_buffers( &
                                          v_vf, 1,-1 )
                
            END IF
            
            IF(bc_x%end <= -3) THEN         ! Ghost-cell extrap. BC at end
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        v_vf(i)%sf(m+j,0:n,0:p) = &
                        v_vf(i)%sf( m ,0:n,0:p)
                    END DO
                END DO
                
            ELSEIF(bc_x%end == -2) THEN     ! Symmetry BC at end
                
                DO j = 1, buff_size
                    
                    DO i = 1, cont_idx%end
                        v_vf(i)%sf(  m+j  ,0:n,0:p) = &
                        v_vf(i)%sf(m-(j-1),0:n,0:p)
                    END DO
                    
                    v_vf(mom_idx%beg)%sf(  m+j  ,0:n,0:p) = &
                   -v_vf(mom_idx%beg)%sf(m-(j-1),0:n,0:p)
                    
                    DO i = mom_idx%beg+1, sys_size
                        v_vf(i)%sf(  m+j  ,0:n,0:p) = &
                        v_vf(i)%sf(m-(j-1),0:n,0:p)
                    END DO
                    
                END DO
                
            ELSEIF(bc_x%end == -1) THEN     ! Periodic BC at end
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        v_vf(i)%sf(m+j,0:n,0:p) = &
                        v_vf(i)%sf(j-1,0:n,0:p)
                    END DO
                END DO
                
            ELSE                            ! Processor BC at end
                
                CALL s_mpi_sendrecv_conservative_variables_buffers( &
                                          v_vf, 1, 1 )
                
            END IF
            
            ! END: Population of Buffers in x-direction ========================
            
            
            ! Population of Buffers in y-direction =============================
            
            IF(n == 0) THEN
                
                RETURN
                
            ELSEIF(bc_y%beg <= -3 .AND. bc_y%beg /= -13) THEN     ! Ghost-cell extrap. BC at beginning
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        v_vf(i)%sf(:,-j,0:p) = &
                        v_vf(i)%sf(:, 0,0:p)
                    END DO
                END DO
                
            ELSEIF(bc_y%beg == -13) THEN    ! Axis BC at beginning

                DO j = 1, buff_size
                    DO k = 0, p
                        IF (z_cc(k) < pi) THEN
                            DO i = 1, mom_idx%beg
                                v_vf(i)%sf(:,-j ,     k     ) = &
                                v_vf(i)%sf(:,j-1,k+((p+1)/2))
                            END DO

                             v_vf(mom_idx%beg+1)%sf(:,-j ,     k     ) = &
                            -v_vf(mom_idx%beg+1)%sf(:,j-1,k+((p+1)/2))
                            
                             v_vf(mom_idx%end)%sf(:,-j ,     k     ) = &
                            -v_vf(mom_idx%end)%sf(:,j-1,k+((p+1)/2))
                            
                            DO i = E_idx, sys_size
                                v_vf(i)%sf(:,-j ,     k     ) = &
                                v_vf(i)%sf(:,j-1,k+((p+1)/2))
                            END DO
                        ELSE
                            DO i = 1, mom_idx%beg
                                v_vf(i)%sf(:,-j ,     k     ) = &
                                v_vf(i)%sf(:,j-1,k-((p+1)/2))
                            END DO

                             v_vf(mom_idx%beg+1)%sf(:,-j ,     k     ) = &
                            -v_vf(mom_idx%beg+1)%sf(:,j-1,k-((p+1)/2))
                            
                             v_vf(mom_idx%end)%sf(:,-j ,     k     ) = &
                            -v_vf(mom_idx%end)%sf(:,j-1,k-((p+1)/2))
                            
                            DO i = E_idx, sys_size
                                v_vf(i)%sf(:,-j ,     k     ) = &
                                v_vf(i)%sf(:,j-1,k-((p+1)/2))
                            END DO
                        END IF
                    END DO
                END DO

            ELSEIF(bc_y%beg == -2) THEN     ! Symmetry BC at beginning
                
                DO j = 1, buff_size
                    
                    DO i = 1, mom_idx%beg
                        v_vf(i)%sf(:,-j ,0:p) = &
                        v_vf(i)%sf(:,j-1,0:p)
                    END DO
                    
                    v_vf(mom_idx%beg+1)%sf(:,-j ,0:p) = &
                   -v_vf(mom_idx%beg+1)%sf(:,j-1,0:p)
                    
                    DO i = mom_idx%beg+2, sys_size
                        v_vf(i)%sf(:,-j ,0:p) = &
                        v_vf(i)%sf(:,j-1,0:p)
                    END DO
                    
                END DO
                
            ELSEIF(bc_y%beg == -1) THEN     ! Periodic BC at beginning
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        v_vf(i)%sf(:,  -j   ,0:p) = &
                        v_vf(i)%sf(:,n-(j-1),0:p)
                    END DO
                END DO
                
            ELSE                            ! Processor BC at beginning
                
                CALL s_mpi_sendrecv_conservative_variables_buffers( &
                                          v_vf, 2,-1 )
                
            END IF
            
            IF(bc_y%end <= -3) THEN         ! Ghost-cell extrap. BC at end
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        v_vf(i)%sf(:,n+j,0:p) = &
                        v_vf(i)%sf(:, n ,0:p)
                    END DO
                END DO
                
            ELSEIF(bc_y%end == -2) THEN     ! Symmetry BC at end
                
                DO j = 1, buff_size
                    
                    DO i = 1, mom_idx%beg
                        v_vf(i)%sf(:,  n+j  ,0:p) = &
                        v_vf(i)%sf(:,n-(j-1),0:p)
                    END DO
                    
                    v_vf(mom_idx%beg+1)%sf(:,  n+j  ,0:p) = &
                   -v_vf(mom_idx%beg+1)%sf(:,n-(j-1),0:p)
                    
                    DO i = mom_idx%beg+2, sys_size
                        v_vf(i)%sf(:,  n+j  ,0:p) = &
                        v_vf(i)%sf(:,n-(j-1),0:p)
                    END DO
                    
                END DO
                
            ELSEIF(bc_y%end == -1) THEN     ! Periodic BC at end
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        v_vf(i)%sf(:,n+j,0:p) = &
                        v_vf(i)%sf(:,j-1,0:p)
                    END DO
                END DO
                
            ELSE                            ! Processor BC at end
                
                CALL s_mpi_sendrecv_conservative_variables_buffers( &
                                          v_vf, 2, 1 )
                
            END IF
            
            ! END: Population of Buffers in y-direction ========================
            
            
            ! Population of Buffers in z-direction =============================
            
            IF(p == 0) THEN
                
                RETURN
                
            ELSEIF(bc_z%beg <= -3) THEN     ! Ghost-cell extrap. BC at beginning
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        v_vf(i)%sf(:,:,-j) = &
                        v_vf(i)%sf(:,:, 0)
                    END DO
                END DO
                
            ELSEIF(bc_z%beg == -2) THEN     ! Symmetry BC at beginning
                
                DO j = 1, buff_size
                    
                    DO i = 1, mom_idx%beg+1
                        v_vf(i)%sf(:,:,-j ) = &
                        v_vf(i)%sf(:,:,j-1)
                    END DO
                    
                    v_vf(mom_idx%end)%sf(:,:,-j ) = &
                   -v_vf(mom_idx%end)%sf(:,:,j-1)
                    
                    DO i = E_idx, sys_size
                        v_vf(i)%sf(:,:,-j ) = &
                        v_vf(i)%sf(:,:,j-1)
                    END DO
                    
                END DO
                
            ELSEIF(bc_z%beg == -1) THEN     ! Periodic BC at beginning
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        v_vf(i)%sf(:,:,  -j   ) = &
                        v_vf(i)%sf(:,:,p-(j-1))
                    END DO
                END DO
                
            ELSE                            ! Processor BC at beginning
                
                CALL s_mpi_sendrecv_conservative_variables_buffers( &
                                          v_vf, 3,-1 )
                
            END IF
            
            IF(bc_z%end <= -3) THEN         ! Ghost-cell extrap. BC at end
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        v_vf(i)%sf(:,:,p+j) = &
                        v_vf(i)%sf(:,:, p )
                    END DO
                END DO
                
            ELSEIF(bc_z%end == -2) THEN     ! Symmetry BC at end
                
                DO j = 1, buff_size
                    
                    DO i = 1, mom_idx%beg+1
                        v_vf(i)%sf(:,:,  p+j  ) = &
                        v_vf(i)%sf(:,:,p-(j-1))
                    END DO
                    
                    v_vf(mom_idx%end)%sf(:,:,  p+j  ) = &
                   -v_vf(mom_idx%end)%sf(:,:,p-(j-1))
                    
                    DO i = E_idx, sys_size
                        v_vf(i)%sf(:,:,  p+j  ) = &
                        v_vf(i)%sf(:,:,p-(j-1))
                    END DO
                    
                END DO
                
            ELSEIF(bc_z%end == -1) THEN     ! Periodic BC at end
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        v_vf(i)%sf(:,:,p+j) = &
                        v_vf(i)%sf(:,:,j-1)
                    END DO
                END DO
                
            ELSE                            ! Processor BC at end
                
                CALL s_mpi_sendrecv_conservative_variables_buffers( &
                                          v_vf, 3, 1 )
                
            END IF
            
            ! END: Population of Buffers in z-direction ========================
            
            
        END SUBROUTINE s_populate_variables_buffers ! -------------





        SUBROUTINE s_populate_conservative_variables_buffers() ! ---------------
        ! Description: The purpose of this procedure is to populate the buffers
        !              of the conservative variables, depending on the selected
        !              boundary conditions.
            
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l,r
            
            
            ! Population of Buffers in x-direction =============================
            
            IF(bc_x%beg <= -3) THEN         ! Ghost-cell extrap. BC at beginning
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        q_cons_qp(0,0,0)%vf(i)%sf(-j,0:n,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf( 0,0:n,0:p)
                    END DO
                END DO
                
            ELSEIF(bc_x%beg == -2) THEN     ! Symmetry BC at beginning
                
                DO j = 1, buff_size
                    
                    DO i = 1, cont_idx%end
                        q_cons_qp(0,0,0)%vf(i)%sf(-j ,0:n,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(j-1,0:n,0:p)
                    END DO
                    
                    q_cons_qp(0,0,0)%vf(mom_idx%beg)%sf(-j ,0:n,0:p) = &
                   -q_cons_qp(0,0,0)%vf(mom_idx%beg)%sf(j-1,0:n,0:p)
                    
                    DO i = mom_idx%beg+1, sys_size
                        q_cons_qp(0,0,0)%vf(i)%sf(-j ,0:n,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(j-1,0:n,0:p)
                    END DO
                    
                END DO
                
            ELSEIF(bc_x%beg == -1) THEN     ! Periodic BC at beginning
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        q_cons_qp(0,0,0)%vf(i)%sf(  -j   ,0:n,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(m-(j-1),0:n,0:p)
                    END DO
                END DO
                
            ELSE                            ! Processor BC at beginning
                
                CALL s_mpi_sendrecv_conservative_variables_buffers( &
                                          q_cons_qp(0,0,0)%vf, 1,-1 )
                
            END IF
            
            IF(bc_x%end <= -3) THEN         ! Ghost-cell extrap. BC at end
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        q_cons_qp(0,0,0)%vf(i)%sf(m+j,0:n,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf( m ,0:n,0:p)
                    END DO
                END DO
                
            ELSEIF(bc_x%end == -2) THEN     ! Symmetry BC at end
                
                DO j = 1, buff_size
                    
                    DO i = 1, cont_idx%end
                        q_cons_qp(0,0,0)%vf(i)%sf(  m+j  ,0:n,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(m-(j-1),0:n,0:p)
                    END DO
                    
                    q_cons_qp(0,0,0)%vf(mom_idx%beg)%sf(  m+j  ,0:n,0:p) = &
                   -q_cons_qp(0,0,0)%vf(mom_idx%beg)%sf(m-(j-1),0:n,0:p)
                    
                    DO i = mom_idx%beg+1, sys_size
                        q_cons_qp(0,0,0)%vf(i)%sf(  m+j  ,0:n,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(m-(j-1),0:n,0:p)
                    END DO
                    
                END DO
                
            ELSEIF(bc_x%end == -1) THEN     ! Periodic BC at end
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        q_cons_qp(0,0,0)%vf(i)%sf(m+j,0:n,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(j-1,0:n,0:p)
                    END DO
                END DO
                
            ELSE                            ! Processor BC at end
                
                CALL s_mpi_sendrecv_conservative_variables_buffers( &
                                          q_cons_qp(0,0,0)%vf, 1, 1 )
                
            END IF
            
            IF(commute_err) THEN
                DO i = ieta%beg, ieta%end, 2
                    DO j = iksi%beg, iksi%end, 2
                        DO k = itau%beg, itau%end, 2
                            DO l = 1, sys_size
                                DO r = 1, buff_size
                                    q_cons_qp(i,j,k)%vf(l)%sf(-r ,0:n,0:p) = &
                                    q_cons_qp(0,0,0)%vf(l)%sf(-r ,0:n,0:p)
                                    q_cons_qp(i,j,k)%vf(l)%sf(m+r,0:n,0:p) = &
                                    q_cons_qp(0,0,0)%vf(l)%sf(m+r,0:n,0:p)
                                END DO
                            END DO
                        END DO
                    END DO
                END DO
            END IF
            
            ! END: Population of Buffers in x-direction ========================
            
            
            ! Population of Buffers in y-direction =============================
            
            IF(n == 0) THEN
                
                RETURN
                
            ELSEIF(bc_y%beg <= -3 .AND. bc_y%beg /= -13) THEN     ! Ghost-cell extrap. BC at beginning
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        q_cons_qp(0,0,0)%vf(i)%sf(:,-j,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:, 0,0:p)
                    END DO
                END DO
                
            ELSEIF(bc_y%beg == -13) THEN    ! Axis BC at beginning

                DO j = 1, buff_size
                    DO k = 0, p
                        IF (z_cc(k) < pi) THEN
                            DO i = 1, mom_idx%beg
                                q_cons_qp(0,0,0)%vf(i)%sf(:,-j ,     k     ) = &
                                q_cons_qp(0,0,0)%vf(i)%sf(:,j-1,k+((p+1)/2))
                            END DO

                             q_cons_qp(0,0,0)%vf(mom_idx%beg+1)%sf(:,-j ,     k     ) = &
                            -q_cons_qp(0,0,0)%vf(mom_idx%beg+1)%sf(:,j-1,k+((p+1)/2))
                            
                             q_cons_qp(0,0,0)%vf(mom_idx%end)%sf(:,-j ,     k     ) = &
                            -q_cons_qp(0,0,0)%vf(mom_idx%end)%sf(:,j-1,k+((p+1)/2))
                            
                            DO i = E_idx, sys_size
                                q_cons_qp(0,0,0)%vf(i)%sf(:,-j ,     k     ) = &
                                q_cons_qp(0,0,0)%vf(i)%sf(:,j-1,k+((p+1)/2))
                            END DO
                        ELSE
                            DO i = 1, mom_idx%beg
                                q_cons_qp(0,0,0)%vf(i)%sf(:,-j ,     k     ) = &
                                q_cons_qp(0,0,0)%vf(i)%sf(:,j-1,k-((p+1)/2))
                            END DO

                             q_cons_qp(0,0,0)%vf(mom_idx%beg+1)%sf(:,-j ,     k     ) = &
                            -q_cons_qp(0,0,0)%vf(mom_idx%beg+1)%sf(:,j-1,k-((p+1)/2))
                            
                             q_cons_qp(0,0,0)%vf(mom_idx%end)%sf(:,-j ,     k     ) = &
                            -q_cons_qp(0,0,0)%vf(mom_idx%end)%sf(:,j-1,k-((p+1)/2))
                            
                            DO i = E_idx, sys_size
                                q_cons_qp(0,0,0)%vf(i)%sf(:,-j ,     k     ) = &
                                q_cons_qp(0,0,0)%vf(i)%sf(:,j-1,k-((p+1)/2))
                            END DO
                        END IF
                    END DO
                END DO

            ELSEIF(bc_y%beg == -2) THEN     ! Symmetry BC at beginning
                
                DO j = 1, buff_size
                    
                    DO i = 1, mom_idx%beg
                        q_cons_qp(0,0,0)%vf(i)%sf(:,-j ,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,j-1,0:p)
                    END DO
                    
                    q_cons_qp(0,0,0)%vf(mom_idx%beg+1)%sf(:,-j ,0:p) = &
                   -q_cons_qp(0,0,0)%vf(mom_idx%beg+1)%sf(:,j-1,0:p)
                    
                    DO i = mom_idx%beg+2, sys_size
                        q_cons_qp(0,0,0)%vf(i)%sf(:,-j ,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,j-1,0:p)
                    END DO
                    
                END DO
                
            ELSEIF(bc_y%beg == -1) THEN     ! Periodic BC at beginning
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        q_cons_qp(0,0,0)%vf(i)%sf(:,  -j   ,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,n-(j-1),0:p)
                    END DO
                END DO
                
            ELSE                            ! Processor BC at beginning
                
                CALL s_mpi_sendrecv_conservative_variables_buffers( &
                                          q_cons_qp(0,0,0)%vf, 2,-1 )
                
            END IF
            
            IF(bc_y%end <= -3) THEN         ! Ghost-cell extrap. BC at end
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        q_cons_qp(0,0,0)%vf(i)%sf(:,n+j,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:, n ,0:p)
                    END DO
                END DO
                
            ELSEIF(bc_y%end == -2) THEN     ! Symmetry BC at end
                
                DO j = 1, buff_size
                    
                    DO i = 1, mom_idx%beg
                        q_cons_qp(0,0,0)%vf(i)%sf(:,  n+j  ,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,n-(j-1),0:p)
                    END DO
                    
                    q_cons_qp(0,0,0)%vf(mom_idx%beg+1)%sf(:,  n+j  ,0:p) = &
                   -q_cons_qp(0,0,0)%vf(mom_idx%beg+1)%sf(:,n-(j-1),0:p)
                    
                    DO i = mom_idx%beg+2, sys_size
                        q_cons_qp(0,0,0)%vf(i)%sf(:,  n+j  ,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,n-(j-1),0:p)
                    END DO
                    
                END DO
                
            ELSEIF(bc_y%end == -1) THEN     ! Periodic BC at end
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        q_cons_qp(0,0,0)%vf(i)%sf(:,n+j,0:p) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,j-1,0:p)
                    END DO
                END DO
                
            ELSE                            ! Processor BC at end
                
                CALL s_mpi_sendrecv_conservative_variables_buffers( &
                                          q_cons_qp(0,0,0)%vf, 2, 1 )
                
            END IF
            
            IF(commute_err) THEN
                DO i = ieta%beg, ieta%end, 2
                    DO j = iksi%beg, iksi%end, 2
                        DO k = itau%beg, itau%end, 2
                            DO l = 1, sys_size
                                DO r = 1, buff_size
                                    q_cons_qp(i,j,k)%vf(l)%sf(:,-r ,0:p) = &
                                    q_cons_qp(0,0,0)%vf(l)%sf(:,-r ,0:p)
                                    q_cons_qp(i,j,k)%vf(l)%sf(:,n+r,0:p) = &
                                    q_cons_qp(0,0,0)%vf(l)%sf(:,n+r,0:p)
                                END DO
                            END DO
                        END DO
                    END DO
                END DO
            END IF
            
            ! END: Population of Buffers in y-direction ========================
            
            
            ! Population of Buffers in z-direction =============================
            
            IF(p == 0) THEN
                
                RETURN
                
            ELSEIF(bc_z%beg <= -3) THEN     ! Ghost-cell extrap. BC at beginning
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,-j) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:, 0)
                    END DO
                END DO
                
            ELSEIF(bc_z%beg == -2) THEN     ! Symmetry BC at beginning
                
                DO j = 1, buff_size
                    
                    DO i = 1, mom_idx%beg+1
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,-j ) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,j-1)
                    END DO
                    
                    q_cons_qp(0,0,0)%vf(mom_idx%end)%sf(:,:,-j ) = &
                   -q_cons_qp(0,0,0)%vf(mom_idx%end)%sf(:,:,j-1)
                    
                    DO i = E_idx, sys_size
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,-j ) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,j-1)
                    END DO
                    
                END DO
                
            ELSEIF(bc_z%beg == -1) THEN     ! Periodic BC at beginning
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,  -j   ) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,p-(j-1))
                    END DO
                END DO
                
            ELSE                            ! Processor BC at beginning
                
                CALL s_mpi_sendrecv_conservative_variables_buffers( &
                                          q_cons_qp(0,0,0)%vf, 3,-1 )
                
            END IF
            
            IF(bc_z%end <= -3) THEN         ! Ghost-cell extrap. BC at end
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,p+j) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:, p )
                    END DO
                END DO
                
            ELSEIF(bc_z%end == -2) THEN     ! Symmetry BC at end
                
                DO j = 1, buff_size
                    
                    DO i = 1, mom_idx%beg+1
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,  p+j  ) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,p-(j-1))
                    END DO
                    
                    q_cons_qp(0,0,0)%vf(mom_idx%end)%sf(:,:,  p+j  ) = &
                   -q_cons_qp(0,0,0)%vf(mom_idx%end)%sf(:,:,p-(j-1))
                    
                    DO i = E_idx, sys_size
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,  p+j  ) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,p-(j-1))
                    END DO
                    
                END DO
                
            ELSEIF(bc_z%end == -1) THEN     ! Periodic BC at end
                
                DO i = 1, sys_size
                    DO j = 1, buff_size
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,p+j) = &
                        q_cons_qp(0,0,0)%vf(i)%sf(:,:,j-1)
                    END DO
                END DO
                
            ELSE                            ! Processor BC at end
                
                CALL s_mpi_sendrecv_conservative_variables_buffers( &
                                          q_cons_qp(0,0,0)%vf, 3, 1 )
                
            END IF
            
            IF(commute_err) THEN
                DO i = ieta%beg, ieta%end, 2
                    DO j = iksi%beg, iksi%end, 2
                        DO k = itau%beg, itau%end, 2
                            DO l = 1, sys_size
                                DO r = 1, buff_size
                                    q_cons_qp(i,j,k)%vf(l)%sf(:,:,-r ) = &
                                    q_cons_qp(0,0,0)%vf(l)%sf(:,:,-r )
                                    q_cons_qp(i,j,k)%vf(l)%sf(:,:,p+r) = &
                                    q_cons_qp(0,0,0)%vf(l)%sf(:,:,p+r)
                                END DO
                            END DO
                        END DO
                    END DO
                END DO
            END IF
            
            ! END: Population of Buffers in z-direction ========================
            
            
        END SUBROUTINE s_populate_conservative_variables_buffers ! -------------
        
        
        
        
        
        SUBROUTINE s_reconstruct_cell_interior_values(v_qp) ! ------------------
        ! Description: The goal of this procedure is to WENO-reconstruct the
        !              inputted cell-averaged variables at the cell-interior
        !              Gaussian quadrature points.
            
            
            ! Inputted cell-averaged variables and their WENO-reconstructed
            ! values at the chosen cell-interior Gaussian quadrature points
            TYPE(vector_field),             &
            DIMENSION( ieta%beg:ieta%end,   &
                       iksi%beg:iksi%end,   &
                       itau%beg:itau%end ), &
            INTENT(INOUT) :: v_qp
            
            ! Indical bounds in the s1-, s2- and s3-directions
            TYPE(bounds_info) :: is1,is2,is3
            
            
            ! Reconstruction in x-direction ====================================
            IF(commute_err .NEQV. .TRUE.) RETURN
            
            is1 = ix; is2 = iy; is3 = iz
            
            is1%beg = is1%beg + weno_polyn; is1%end = is1%end - weno_polyn
            
            CALL s_weno( v_qp( 0, 0, 0)%vf(iv%beg:iv%end), &
                         v_qp(-1, 0, 0)%vf(iv%beg:iv%end), &
                         v_qp( 1, 0, 0)%vf(iv%beg:iv%end), &
                         dflt_int, 1, 1, 2, is1,is2,is3    )
            ! ==================================================================
            
            
            ! Reconstruction in y-direction ====================================
            IF(n == 0) RETURN
            
            is2%beg = is2%beg + weno_polyn; is2%end = is2%end - weno_polyn
            
            CALL s_weno( v_qp(-1, 0, 0)%vf(iv%beg:iv%end), &
                         v_qp(-1,-1, 0)%vf(iv%beg:iv%end), &
                         v_qp(-1, 1, 0)%vf(iv%beg:iv%end), &
                         dflt_int, 1, 2, 2, is1,is2,is3    )
            CALL s_weno( v_qp( 1, 0, 0)%vf(iv%beg:iv%end), &
                         v_qp( 1,-1, 0)%vf(iv%beg:iv%end), &
                         v_qp( 1, 1, 0)%vf(iv%beg:iv%end), &
                         dflt_int, 1, 2, 2, is1,is2,is3    )
            ! ==================================================================
            
            
            ! Reconstruction in z-direction ====================================
            IF(p == 0) RETURN
            
            is3%beg = is3%beg + weno_polyn; is3%end = is3%end - weno_polyn
            
            CALL s_weno( v_qp(-1,-1, 0)%vf(iv%beg:iv%end), &
                         v_qp(-1,-1,-1)%vf(iv%beg:iv%end), &
                         v_qp(-1,-1, 1)%vf(iv%beg:iv%end), &
                         dflt_int, 1, 3, 2, is1,is2,is3    )
            CALL s_weno( v_qp(-1, 1, 0)%vf(iv%beg:iv%end), &
                         v_qp(-1, 1,-1)%vf(iv%beg:iv%end), &
                         v_qp(-1, 1, 1)%vf(iv%beg:iv%end), &
                         dflt_int, 1, 3, 2, is1,is2,is3    )
            CALL s_weno( v_qp( 1,-1, 0)%vf(iv%beg:iv%end), &
                         v_qp( 1,-1,-1)%vf(iv%beg:iv%end), &
                         v_qp( 1,-1, 1)%vf(iv%beg:iv%end), &
                         dflt_int, 1, 3, 2, is1,is2,is3    )
            CALL s_weno( v_qp( 1, 1, 0)%vf(iv%beg:iv%end), &
                         v_qp( 1, 1,-1)%vf(iv%beg:iv%end), &
                         v_qp( 1, 1, 1)%vf(iv%beg:iv%end), &
                         dflt_int, 1, 3, 2, is1,is2,is3    )
            ! ==================================================================
            
            
        END SUBROUTINE s_reconstruct_cell_interior_values ! --------------------
        
        
        
        
        
        SUBROUTINE s_average_cell_interior_values(v_qp) ! ----------------------
        ! Description: The goal of this subroutine is to numerically approximate
        !              the cell-interior integral-average of the given variables
        !              by taking the arithmetic mean of their WENO-reconstructed
        !              values at the cell-interior Gaussian quadrature points.
            
            
            ! The inputted WENO-reconstructed values of cell-averaged variables
            ! at cell-interior Gaussian quadrature points and their numerically
            ! approximated cell-interior integral-average.
            TYPE(vector_field),             &
            DIMENSION( ieta%beg:ieta%end,   &
                       iksi%beg:iksi%end,   &
                       itau%beg:itau%end ), &
            INTENT(INOUT) :: v_qp
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l
            
            
            IF(commute_err .NEQV. .TRUE.) RETURN
            
            DO l = iv%beg, iv%end
                
                v_qp(0,0,0)%vf(l)%sf = 0d0
                
                DO k = itau%beg, itau%end, 2
                    DO j = iksi%beg, iksi%end, 2
                        DO i = ieta%beg, ieta%end, 2
                            v_qp(0,0,0)%vf(l)%sf = v_qp(0,0,0)%vf(l)%sf &
                                                 + v_qp(i,j,k)%vf(l)%sf
                        END DO
                    END DO
                END DO
                
                v_qp(0,0,0)%vf(l)%sf = v_qp(0,0,0)%vf(l)%sf / (2**num_dims)
                
            END DO
            
            
        END SUBROUTINE s_average_cell_interior_values ! ------------------------
        
        
        
        
        
        SUBROUTINE s_reconstruct_cell_boundary_values( v_vf, vL_qp, vR_qp, & ! -
                                                        cd_vars, norm_dir  )
        ! Description: The purpose of this subroutine is to WENO-reconstruct the
        !              left and the right cell-boundary values, including values
        !              at the Gaussian quadrature points, from the cell-averaged
        !              variables.
            
            
            ! Cell-average variables
            TYPE(scalar_field), DIMENSION(iv%beg:iv%end), INTENT(IN) :: v_vf
            
            ! Left and right, WENO-reconstructed, cell-boundary values including
            ! the values at the quadrature points, of the cell-average variables
            TYPE(vector_field),             &
            DIMENSION( ichi%beg:ichi%end,   &
                       ipsi%beg:ipsi%end ), &
            INTENT(INOUT) :: vL_qp, vR_qp
            
            ! Characteristic decomposition state variables type
            INTEGER, INTENT(IN) :: cd_vars
            
            ! Splitting coordinate direction
            INTEGER, INTENT(IN) :: norm_dir
            
            ! Coordinate direction of the WENO reconstruction
            INTEGER :: weno_dir
            
            ! Indical bounds in the s1-, s2- and s3-directions
            TYPE(bounds_info) :: is1,is2,is3
            
            
            ! Reconstruction in s1-direction ===================================
            is1 = ix; is2 = iy; is3 = iz
            
            IF(norm_dir == 1) THEN
                weno_dir = 1; is1%beg = is1%beg + weno_polyn
                              is1%end = is1%end - weno_polyn
            ELSEIF(norm_dir == 2) THEN
                weno_dir = 2; is2%beg = is2%beg + weno_polyn
                              is2%end = is2%end - weno_polyn
            ELSE
                weno_dir = 3; is3%beg = is3%beg + weno_polyn
                              is3%end = is3%end - weno_polyn
            END IF
            
            CALL s_weno(            v_vf(iv%beg:iv%end), &
                         vL_qp( 0, 0)%vf(iv%beg:iv%end), &
                         vR_qp( 0, 0)%vf(iv%beg:iv%end), &
                         cd_vars, norm_dir, weno_dir, 1, &
                         is1,is2,is3                     )
            ! ==================================================================
            !print '(a)', 'SHB: finished weno' 
            
            ! Reconstruction in s2-direction ===================================
            IF(split_err .NEQV. .TRUE.) RETURN
            
            IF(norm_dir /= 1) THEN
                weno_dir = 1; is1%beg = is1%beg + weno_polyn
                              is1%end = is1%end - weno_polyn
            ELSE
                weno_dir = 2; is2%beg = is2%beg + weno_polyn
                              is2%end = is2%end - weno_polyn
            END IF
            
            CALL s_weno( vL_qp( 0, 0)%vf(iv%beg:iv%end), &
                         vL_qp(-1, 0)%vf(iv%beg:iv%end), &
                         vL_qp( 1, 0)%vf(iv%beg:iv%end), &
                         dflt_int, norm_dir,weno_dir, 2, &
                         is1,is2,is3                     )
            CALL s_weno( vR_qp( 0, 0)%vf(iv%beg:iv%end), &
                         vR_qp(-1, 0)%vf(iv%beg:iv%end), &
                         vR_qp( 1, 0)%vf(iv%beg:iv%end), &
                         dflt_int, norm_dir,weno_dir, 2, &
                         is1,is2,is3                     )
            ! ==================================================================
            
            
            ! Reconstruction in s3-direction ===================================
            IF(p == 0) RETURN
            
            IF(norm_dir /= 3) THEN
                weno_dir = 3; is3%beg = is3%beg + weno_polyn
                              is3%end = is3%end - weno_polyn
            ELSE
                weno_dir = 2; is2%beg = is2%beg + weno_polyn
                              is2%end = is2%end - weno_polyn
            END IF
            
            CALL s_weno( vL_qp(-1, 0)%vf(iv%beg:iv%end), &
                         vL_qp(-1,-1)%vf(iv%beg:iv%end), &
                         vL_qp(-1, 1)%vf(iv%beg:iv%end), &
                         dflt_int, norm_dir,weno_dir, 2, &
                         is1,is2,is3                     )
            CALL s_weno( vL_qp( 1, 0)%vf(iv%beg:iv%end), &
                         vL_qp( 1,-1)%vf(iv%beg:iv%end), &
                         vL_qp( 1, 1)%vf(iv%beg:iv%end), &
                         dflt_int, norm_dir,weno_dir, 2, &
                         is1,is2,is3                     )
            CALL s_weno( vR_qp(-1, 0)%vf(iv%beg:iv%end), &
                         vR_qp(-1,-1)%vf(iv%beg:iv%end), &
                         vR_qp(-1, 1)%vf(iv%beg:iv%end), &
                         dflt_int, norm_dir,weno_dir, 2, &
                         is1,is2,is3                     )
            CALL s_weno( vR_qp( 1, 0)%vf(iv%beg:iv%end), &
                         vR_qp( 1,-1)%vf(iv%beg:iv%end), &
                         vR_qp( 1, 1)%vf(iv%beg:iv%end), &
                         dflt_int, norm_dir,weno_dir, 2, &
                         is1,is2,is3                     )
            ! ==================================================================
            
            
        END SUBROUTINE s_reconstruct_cell_boundary_values ! --------------------
        
        
        
        
        
        SUBROUTINE s_average_cell_boundary_values(vK_qp) ! ---------------------
        ! Description: The goal of the procedure is to numerically approximate
        !              the left or right cell-boundary integral-average of the
        !              given variables by getting the arithmetic mean of their
        !              WENO-reconstructed values at the cell-boundary Gaussian
        !              quadrature points.
            
            
            ! The inputted WENO-reconstructed values of cell-averaged variables
            ! at the left or right cell-boundary Gaussian quadrature points and
            ! their numerically approximated cell-boundary integral-average.
            TYPE(vector_field),             &
            DIMENSION( ichi%beg:ichi%end,   &
                       ipsi%beg:ipsi%end ), &
            INTENT(INOUT) :: vK_qp
            
            ! Generic loop iterators
            INTEGER :: i,j,k
            
            
            IF(split_err .NEQV. .TRUE.) RETURN
            
            DO k = iv%beg, iv%end
                
                vK_qp(0,0)%vf(k)%sf = 0d0
                
                DO j = ipsi%beg, ipsi%end, 2
                    DO i = ichi%beg, ichi%end, 2
                        vK_qp(0,0)%vf(k)%sf = vK_qp(0,0)%vf(k)%sf &
                                            + vK_qp(i,j)%vf(k)%sf
                    END DO
                END DO
                
                vK_qp(0,0)%vf(k)%sf = vK_qp(0,0)%vf(k)%sf / (2**(num_dims - 1))
                
            END DO
            
            
        END SUBROUTINE s_average_cell_boundary_values ! ------------------------
        
        
        
        
        
        SUBROUTINE s_apply_scalar_divergence_theorem( vL_vf, vR_vf, & ! --------
                                                          dv_ds_vf, &
                                                          norm_dir  )
        ! Description: The purpose of this subroutine is to employ the inputted
        !              left and right cell-boundary integral-averaged variables
        !              to compute the relevant cell-average first-order spatial
        !              derivatives in the x-, y- or z-direction by means of the
        !              scalar divergence theorem.
            
            
            ! Left and right cell-boundary integral-averages
            TYPE(scalar_field),       &
            DIMENSION(iv%beg:iv%end), &
            INTENT(IN) :: vL_vf, vR_vf
            
            ! Cell-average first-order spatial derivatives
            TYPE(scalar_field),       &
            DIMENSION(iv%beg:iv%end), &
            INTENT(INOUT) :: dv_ds_vf
            
            ! Splitting coordinate direction
            INTEGER, INTENT(IN) :: norm_dir
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l
            
            
            ! First-Order Spatial Derivatives in x-direction ===================
            IF(norm_dir == 1) THEN
                
                ! A general application of the scalar divergence theorem that
                ! utilizes the left and right cell-boundary integral-averages,
                ! inside each cell, or an arithmetic mean of these two at the
                ! cell-boundaries, to calculate the cell-averaged first-order
                ! spatial derivatives inside the cell.
                DO i = iv%beg, iv%end
                    DO l = iz%beg, iz%end
                        DO k = iy%beg, iy%end
                            DO j = ix%beg+1, ix%end-1
                                dv_ds_vf(i)%sf(j,k,l) = &
                                             1d0/((1d0+wa_flg)*dx(j)) &
                                       *( wa_flg*vL_vf(i)%sf(j+1,k,l) &
                                        +        vR_vf(i)%sf( j ,k,l) &
                                        -        vL_vf(i)%sf( j ,k,l) &
                                        - wa_flg*vR_vf(i)%sf(j-1,k,l) )
                            END DO
                        END DO
                    END DO
                END DO
                
                ! Modifying the way the scalar divergence theorem is utilized
                ! near beginning and end Riemann state extrapolation boundary
                ! conditions, respectively, as to omit any contributions from
                ! outside the physical computational domain in the calculated
                ! cell-averaged first-order spatial derivatives
                IF(weno_avg .NEQV. .TRUE.) RETURN
                
                IF(bc_x%beg == -4) THEN     ! Riemann state extrap. BC at beg.
                    DO i = iv%beg, iv%end
                        DO l = iz%beg, iz%end
                            DO k = iy%beg, iy%end
                                dv_ds_vf(i)%sf(0,k,l) = 1d0/dx(0) &
                                            *( vR_vf(i)%sf(0,k,l) &
                                             - vL_vf(i)%sf(0,k,l) )
                            END DO
                        END DO
                    END DO
                END IF
                
                IF(bc_x%end == -4) THEN     ! Riemann state extrap. BC at end
                    DO i = iv%beg, iv%end
                        DO l = iz%beg, iz%end
                            DO k = iy%beg, iy%end
                                dv_ds_vf(i)%sf(m,k,l) = 1d0/dx(m) &
                                            *( vR_vf(i)%sf(m,k,l) &
                                             - vL_vf(i)%sf(m,k,l) )
                            END DO
                        END DO
                    END DO
                END IF
            ! END: First-Order Spatial Derivatives in x-direction ==============
            
            
            ! First-Order Spatial Derivatives in y-direction ===================
            ELSEIF(norm_dir == 2) THEN
                
                ! A general application of the scalar divergence theorem that
                ! utilizes the left and right cell-boundary integral-averages,
                ! inside each cell, or an arithmetic mean of these two at the
                ! cell-boundaries, to calculate the cell-averaged first-order
                ! spatial derivatives inside the cell.
                DO i = iv%beg, iv%end
                    DO l = iz%beg, iz%end
                        DO k = iy%beg+1, iy%end-1
                            DO j = ix%beg, ix%end
                                dv_ds_vf(i)%sf(j,k,l) = &
                                             1d0/((1d0+wa_flg)*dy(k)) &
                                       *( wa_flg*vL_vf(i)%sf(j,k+1,l) &
                                        +        vR_vf(i)%sf(j, k ,l) &
                                        -        vL_vf(i)%sf(j, k ,l) &
                                        - wa_flg*vR_vf(i)%sf(j,k-1,l) )
                            END DO
                        END DO
                    END DO
                END DO
                
                ! Modifying the way the scalar divergence theorem is utilized
                ! near beginning and end Riemann state extrapolation boundary
                ! conditions, respectively, as to omit any contributions from
                ! outside the physical computational domain in the calculated
                ! cell-averaged first-order spatial derivatives
                IF(weno_avg .NEQV. .TRUE.) RETURN
                
                IF(bc_y%beg == -4) THEN     ! Riemann state extrap. BC at beg.
                    DO i = iv%beg, iv%end
                        DO l = iz%beg, iz%end
                            DO j = ix%beg, ix%end
                                dv_ds_vf(i)%sf(j,0,l) = 1d0/dy(0) &
                                            *( vR_vf(i)%sf(j,0,l) &
                                             - vL_vf(i)%sf(j,0,l) )
                            END DO
                        END DO
                    END DO
                END IF
                
                IF(bc_y%end == -4) THEN     ! Riemann state extrap. BC at end
                    DO i = iv%beg, iv%end
                        DO l = iz%beg, iz%end
                            DO j = ix%beg, ix%end
                                dv_ds_vf(i)%sf(j,n,l) = 1d0/dy(n) &
                                            *( vR_vf(i)%sf(j,n,l) &
                                             - vL_vf(i)%sf(j,n,l) )
                            END DO
                        END DO
                    END DO
                END IF
            ! END: First-Order Spatial Derivatives in y-direction ==============
            
            
            ! First-Order Spatial Derivatives in z-direction ===================
            ELSE
                
                ! A general application of the scalar divergence theorem that
                ! utilizes the left and right cell-boundary integral-averages,
                ! inside each cell, or an arithmetic mean of these two at the
                ! cell-boundaries, to calculate the cell-averaged first-order
                ! spatial derivatives inside the cell.
                DO i = iv%beg, iv%end
                    DO l = iz%beg+1, iz%end-1
                        DO k = iy%beg, iy%end
                            DO j = ix%beg, ix%end
                                dv_ds_vf(i)%sf(j,k,l) = &
                                             1d0/((1d0+wa_flg)*dz(l)) &
                                       *( wa_flg*vL_vf(i)%sf(j,k,l+1) &
                                        +        vR_vf(i)%sf(j,k, l ) &
                                        -        vL_vf(i)%sf(j,k, l ) &
                                        - wa_flg*vR_vf(i)%sf(j,k,l-1) )
                            END DO
                        END DO
                    END DO
                END DO
                
                ! Modifying the way the scalar divergence theorem is utilized
                ! near beginning and end Riemann state extrapolation boundary
                ! conditions, respectively, as to omit any contributions from
                ! outside the physical computational domain in the calculated
                ! cell-averaged first-order spatial derivatives
                IF(weno_avg .NEQV. .TRUE.) RETURN
                
                IF(bc_z%beg == -4) THEN     ! Riemann state extrap. BC at beg.
                    DO i = iv%beg, iv%end
                        DO k = iy%beg, iy%end
                            DO j = ix%beg, ix%end
                                dv_ds_vf(i)%sf(j,k,0) = 1d0/dz(0) &
                                            *( vR_vf(i)%sf(j,k,0) &
                                             - vL_vf(i)%sf(j,k,0) )
                            END DO
                        END DO
                    END DO
                END IF
                
                IF(bc_z%end == -4) THEN     ! Riemann state extrap. BC at end
                    DO i = iv%beg, iv%end
                        DO k = iy%beg, iy%end
                            DO j = ix%beg, ix%end
                                dv_ds_vf(i)%sf(j,k,p) = 1d0/dz(p) &
                                            *( vR_vf(i)%sf(j,k,p) &
                                             - vL_vf(i)%sf(j,k,p) )
                            END DO
                        END DO
                    END DO
                END IF
                
            END IF
            ! END: First-Order Spatial Derivatives in z-direction ==============
            
            
        END SUBROUTINE s_apply_scalar_divergence_theorem ! ---------------------
        
        
        
        
        
        SUBROUTINE s_apply_vector_divergence_theorem(       & ! ----------------
                           vL_x_ndqp, vL_y_ndqp, vL_z_ndqp, &
                           vR_x_ndqp, vR_y_ndqp, vR_z_ndqp, &
                                                  div_v_vf  )
        ! Description: The goal of this procedure is to utilize the inputted
        !              left and right cell-boundary integral-averaged vector
        !              components in the x-, y-, and z-directions to compute
        !              the vector divergence by using the divergence theorem.
            
            
            ! The left and right cell-boundary integral-average vector
            ! components in the x-, y- and z-directions, respectively.
            TYPE(vector_field),             &
            DIMENSION(     1   :num_dims,   &
                       ichi%beg:ichi%end,   &
                       ipsi%beg:ipsi%end ), &
            INTENT(IN) :: vL_x_ndqp, vR_x_ndqp, &
                          vL_y_ndqp, vR_y_ndqp, &
                          vL_z_ndqp, vR_z_ndqp
            
            ! Cell-average divergence
            TYPE(scalar_field),       &
            DIMENSION(iv%beg:iv%end), &
            INTENT(INOUT) :: div_v_vf
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l
            
            
            ! First-Order Spatial Derivatives in x-direction ===================
            
            ! General application of the vector divergence theorem which uses
            ! the left and right cell-boundary integral-averages, inside each
            ! cell, or an arithmetic mean of these two at the cell-boundaries,
            ! in order to obtain cell-average first-order spatial derivatives
            ! inside the cell
            DO i = iv%beg, iv%end
                DO l = iz%beg, iz%end
                    DO k = iy%beg, iy%end
                        DO j = ix%beg+1, ix%end-1
                            div_v_vf(i)%sf(j,k,l) = 1d0/((1d0+wa_flg)*dx(j)) &
                                *( wa_flg*vL_x_ndqp(1,0,0)%vf(i)%sf(j+1,k,l) &
                                 +        vR_x_ndqp(1,0,0)%vf(i)%sf( j ,k,l) &
                                 -        vL_x_ndqp(1,0,0)%vf(i)%sf( j ,k,l) &
                                 - wa_flg*vR_x_ndqp(1,0,0)%vf(i)%sf(j-1,k,l) )
                        END DO
                    END DO
                END DO
            END DO
            
            ! Customizing the way the vector divergence theorem is applied near
            ! beginning and end Riemann state extrapolation boundary conditions,
            ! respectively, as to avoid contributions from outside the physical
            ! computational domain in the computation of the cell-average first-
            ! order spatial derivatives
            IF(weno_avg) THEN

                IF(bc_x%beg == -4) THEN     ! Riemann state extrap. BC at beg.
                    DO i = iv%beg, iv%end
                        DO l = iz%beg, iz%end
                            DO k = iy%beg, iy%end
                                div_v_vf(i)%sf(0,k,l) = &
                                       div_v_vf(i)%sf(0,k,l) - 5d-1/dx(0) &
                                    *( vL_x_ndqp(1,0,0)%vf(i)%sf( 1 ,k,l) &
                                     - vR_x_ndqp(1,0,0)%vf(i)%sf( 0 ,k,l) &
                                     + vL_x_ndqp(1,0,0)%vf(i)%sf( 0 ,k,l) &
                                     - vR_x_ndqp(1,0,0)%vf(i)%sf(-1 ,k,l) )
                            END DO
                        END DO
                    END DO
                END IF
                
                IF(bc_x%end == -4) THEN     ! Riemann state extrap. BC at end
                    DO i = iv%beg, iv%end
                        DO l = iz%beg, iz%end
                            DO k = iy%beg, iy%end
                                div_v_vf(i)%sf(m,k,l) = &
                                       div_v_vf(i)%sf(m,k,l) - 5d-1/dx(m) &
                                    *( vL_x_ndqp(1,0,0)%vf(i)%sf(m+1,k,l) &
                                     - vR_x_ndqp(1,0,0)%vf(i)%sf( m ,k,l) &
                                     + vL_x_ndqp(1,0,0)%vf(i)%sf( m ,k,l) &
                                     - vR_x_ndqp(1,0,0)%vf(i)%sf(m-1,k,l) )
                            END DO
                        END DO
                    END DO
                END IF
                
            END IF
            
            ! END: First-Order Spatial Derivatives in x-direction ==============
            
            
            ! First-Order Spatial Derivatives in y-direction ===================
            
            ! General application of the vector divergence theorem which uses
            ! the left and right cell-boundary integral-averages, inside each
            ! cell, or an arithmetic mean of these two at the cell-boundaries,
            ! in order to obtain cell-average first-order spatial derivatives
            ! inside the cell
            IF(n == 0) RETURN
            
            DO i = iv%beg, iv%end
                DO l = iz%beg, iz%end
                    DO k = iy%beg+1, iy%end-1
                        DO j = ix%beg, ix%end
                            div_v_vf(i)%sf(j,k,l) =    div_v_vf(i)%sf(j,k,l) &
                                                  + 1d0/((1d0+wa_flg)*dy(k)) &
                                *( wa_flg*vL_y_ndqp(2,0,0)%vf(i)%sf(j,k+1,l) &
                                 +        vR_y_ndqp(2,0,0)%vf(i)%sf(j, k ,l) &
                                 -        vL_y_ndqp(2,0,0)%vf(i)%sf(j, k ,l) &
                                 - wa_flg*vR_y_ndqp(2,0,0)%vf(i)%sf(j,k-1,l) )
                        END DO
                    END DO
                END DO
            END DO
            
            ! Customizing the way the vector divergence theorem is applied near
            ! beginning and end Riemann state extrapolation boundary conditions,
            ! respectively, as to avoid contributions from outside the physical
            ! computational domain in the computation of the cell-average first-
            ! order spatial derivatives
            IF(weno_avg) THEN

                IF(bc_y%beg == -4) THEN     ! Riemann state extrap. BC at beg.
                    DO i = iv%beg, iv%end
                        DO l = iz%beg, iz%end
                            DO j = ix%beg, ix%end
                                div_v_vf(i)%sf(j,0,l) = &
                                       div_v_vf(i)%sf(j,0,l) - 5d-1/dy(0) &
                                    *( vL_y_ndqp(2,0,0)%vf(i)%sf(j, 1 ,l) &
                                     - vR_y_ndqp(2,0,0)%vf(i)%sf(j, 0 ,l) &
                                     + vL_y_ndqp(2,0,0)%vf(i)%sf(j, 0 ,l) &
                                     - vR_y_ndqp(2,0,0)%vf(i)%sf(j,-1 ,l) )
                            END DO
                        END DO
                    END DO
                END IF
                
                IF(bc_y%end == -4) THEN     ! Riemann state extrap. BC at end
                    DO i = iv%beg, iv%end
                        DO l = iz%beg, iz%end
                            DO j = ix%beg, ix%end
                                div_v_vf(i)%sf(j,n,l) = &
                                       div_v_vf(i)%sf(j,n,l) - 5d-1/dy(n) &
                                    *( vL_y_ndqp(2,0,0)%vf(i)%sf(j,n+1,l) &
                                     - vR_y_ndqp(2,0,0)%vf(i)%sf(j, n ,l) &
                                     + vL_y_ndqp(2,0,0)%vf(i)%sf(j, n ,l) &
                                     - vR_y_ndqp(2,0,0)%vf(i)%sf(j,n-1,l) )
                            END DO
                        END DO
                    END DO
                END IF
                
            END IF
            
            ! END: First-Order Spatial Derivatives in y-direction ==============
            
            
            ! First-Order Spatial Derivatives in z-direction ===================
            
            ! General application of the vector divergence theorem which uses
            ! the left and right cell-boundary integral-averages, inside each
            ! cell, or an arithmetic mean of these two at the cell-boundaries,
            ! in order to obtain cell-average first-order spatial derivatives
            ! inside the cell
            IF(p == 0) RETURN
            
            DO i = iv%beg, iv%end
                DO l = iz%beg+1, iz%end-1
                    DO k = iy%beg, iy%end
                        DO j = ix%beg, ix%end
                            div_v_vf(i)%sf(j,k,l) =    div_v_vf(i)%sf(j,k,l) &
                                                  + 1d0/((1d0+wa_flg)*dz(l)) &
                                *( wa_flg*vL_z_ndqp(3,0,0)%vf(i)%sf(j,k,l+1) &
                                 +        vR_z_ndqp(3,0,0)%vf(i)%sf(j,k, l ) &
                                 -        vL_z_ndqp(3,0,0)%vf(i)%sf(j,k, l ) &
                                 - wa_flg*vR_z_ndqp(3,0,0)%vf(i)%sf(j,k,l-1) )
                        END DO
                    END DO
                END DO
            END DO
            
            ! Customizing the way the vector divergence theorem is applied near
            ! beginning and end Riemann state extrapolation boundary conditions,
            ! respectively, as to avoid contributions from outside the physical
            ! computational domain in the computation of the cell-average first-
            ! order spatial derivatives
            IF(weno_avg) THEN

                IF(bc_z%beg == -4) THEN     ! Riemann state extrap. BC at beg.
                    DO i = iv%beg, iv%end
                        DO k = iy%beg, iy%end
                            DO j = ix%beg, ix%end
                                div_v_vf(i)%sf(j,k,0) = &
                                       div_v_vf(i)%sf(j,k,0) - 5d-1/dz(0) &
                                    *( vL_z_ndqp(3,0,0)%vf(i)%sf(j,k, 1 ) &
                                     - vR_z_ndqp(3,0,0)%vf(i)%sf(j,k, 0 ) &
                                     + vL_z_ndqp(3,0,0)%vf(i)%sf(j,k, 0 ) &
                                     - vR_z_ndqp(3,0,0)%vf(i)%sf(j,k,-1 ) )
                            END DO
                        END DO
                    END DO
                END IF
                
                IF(bc_z%end == -4) THEN     ! Riemann state extrap. BC at end
                    DO i = iv%beg, iv%end
                        DO k = iy%beg, iy%end
                            DO j = ix%beg, ix%end
                                div_v_vf(i)%sf(j,k,p) = &
                                       div_v_vf(i)%sf(j,k,p) - 5d-1/dz(p) &
                                    *( vL_z_ndqp(3,0,0)%vf(i)%sf(j,k,p+1) &
                                     - vR_z_ndqp(3,0,0)%vf(i)%sf(j,k, p ) &
                                     + vL_z_ndqp(3,0,0)%vf(i)%sf(j,k, p ) &
                                     - vR_z_ndqp(3,0,0)%vf(i)%sf(j,k,p-1) )
                            END DO
                        END DO
                    END DO
                END IF
                
            END IF
            
            ! END: First-Order Spatial Derivatives in z-direction ==============
            
            
        END SUBROUTINE s_apply_vector_divergence_theorem ! ---------------------
        
        
        
        
        
        SUBROUTINE s_compute_gradient_magnitude( dv_dx_vf, & ! -----------------
                                                 dv_dy_vf, &
                                                 dv_dz_vf, &
                                                  gm_v_vf  )
        ! Description: The purpose of the procedure is to utilize the inputted
        !              cell-averaged first-order spatial derivatives in the x-,
        !              y- and z-directions to calculate the gradient magnitude.
            
            
            ! Cell-average first-order spatial derivatives in x-,y-,z-directions
            TYPE(scalar_field),       &
            DIMENSION(iv%beg:iv%end), &
            INTENT(IN) :: dv_dx_vf, &
                          dv_dy_vf, &
                          dv_dz_vf
            
            ! Gradient magnitude
            TYPE(scalar_field),       &
            DIMENSION(iv%beg:iv%end), &
            INTENT(INOUT) :: gm_v_vf
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l
            
            ! Scalar Product Contribution in x-direction =======================
            DO i = iv%beg, iv%end
                DO l = iz%beg, iz%end
                    DO k = iy%beg, iy%end
                        DO j = ix%beg, ix%end
                            gm_v_vf(i)%sf(j,k,l) = dv_dx_vf(i)%sf(j,k,l) &
                                                 * dv_dx_vf(i)%sf(j,k,l)
                        END DO
                    END DO
                END DO
            END DO
            ! ==================================================================
            
            
            ! Scalar Product Contribution in y-direction =======================
            IF(n > 0) THEN
               
                DO i = iv%beg, iv%end
                    DO l = iz%beg, iz%end
                        DO k = iy%beg, iy%end
                            DO j = ix%beg, ix%end
                                gm_v_vf(i)%sf(j,k,l) =  gm_v_vf(i)%sf(j,k,l) &
                                                     + dv_dy_vf(i)%sf(j,k,l) &
                                                     * dv_dy_vf(i)%sf(j,k,l)
                            END DO
                        END DO
                    END DO
                END DO
            ! ==================================================================
            
            
            ! Scalar Product Contribution in z-direction =======================
                IF(p > 0) THEN
                    DO i = iv%beg, iv%end
                        DO l = iz%beg, iz%end
                            DO k = iy%beg, iy%end
                                DO j = ix%beg, ix%end
                                    gm_v_vf(i)%sf(j,k,l) =  gm_v_vf(i)%sf(j,k,l) &
                                                         + dv_dz_vf(i)%sf(j,k,l) &
                                                         * dv_dz_vf(i)%sf(j,k,l)
                                END DO
                            END DO
                        END DO
                    END DO
                END IF
               
            END IF
            ! ==================================================================
            
            
            ! Square Root of the Scalar Product ================================
            DO i = iv%beg, iv%end
                DO l = iz%beg, iz%end
                    DO k = iy%beg, iy%end
                        DO j = ix%beg, ix%end
                            gm_v_vf(i)%sf(j,k,l) = SQRT(gm_v_vf(i)%sf(j,k,l))
                        END DO
                    END DO
                END DO
            END DO
            ! ==================================================================
            
            
        END SUBROUTINE s_compute_gradient_magnitude ! --------------------------
        
        
        
        
        
        SUBROUTINE s_finalize_rhs_module() ! -----------------------------------
        ! Description: Module deallocation and/or disassociation procedures
            
            
            ! Generic loop iterators
            INTEGER :: i,j,k,l
            
            
            ! Deallocation/Disassociation of q_cons_qp and q_prim_qp ===========
            DO k = itau%beg, itau%end
                DO j = iksi%beg, iksi%end
                    DO i = ieta%beg, ieta%end
                     
                        IF(i /= 0 .AND. ABS(j) >= ABS(k)) THEN
                        
                            DO l = 1, sys_size
                                DEALLOCATE(q_cons_qp(i,j,k)%vf(l)%sf)
                            END DO
                        
                            DO l = mom_idx%beg, E_idx
                                DEALLOCATE(q_prim_qp(i,j,k)%vf(l)%sf)
                            END DO

                            IF (model_eqns == 3) THEN
                                DO l = internalEnergies_idx%beg, internalEnergies_idx%end
                                    DEALLOCATE(q_prim_qp(i,j,k)%vf(l)%sf)
                                END DO
                            END IF
                        
                            DO l = 1, sys_size
                                NULLIFY(q_prim_qp(i,j,k)%vf(l)%sf)
                            END DO
                        
                        END IF
                     
                        DEALLOCATE(q_cons_qp(i,j,k)%vf, q_prim_qp(i,j,k)%vf)
                     
                    END DO
                END DO
            END DO
            
            DEALLOCATE(q_cons_qp, q_prim_qp)
            ! ==================================================================
            
            
            ! Deallocation/Disassociation of qK_cons_ndqp and qK_prim_ndqp =====
            DO k = ipsi%beg, ipsi%end
                DO j = ichi%beg, ichi%end
                    DO i = num_dims, 1, -1
                     
                        IF(ABS(j) >= ABS(k)) THEN
                        
                            DO l = 1, cont_idx%end
                                NULLIFY(qL_prim_ndqp(i,j,k)%vf(l)%sf)
                                NULLIFY(qR_prim_ndqp(i,j,k)%vf(l)%sf)
                            END DO
                        
                            DO l = adv_idx%beg, adv_idx%end
                                NULLIFY(qL_prim_ndqp(i,j,k)%vf(l)%sf)
                                NULLIFY(qR_prim_ndqp(i,j,k)%vf(l)%sf)
                            END DO
                            
                            IF(i /= 1) THEN
                               
                                IF( (char_decomp .NEQV. .TRUE.) &
                                               .AND.            &
                                        ANY(Re_size > 0)       ) THEN
                                    IF(weno_vars == 1) THEN
                                        DO l = 1, mom_idx%end
                                            DEALLOCATE(qL_cons_ndqp(i,j,k)%vf(l)%sf)
                                            DEALLOCATE(qR_cons_ndqp(i,j,k)%vf(l)%sf)
                                        END DO
                                    ELSE
                                        DO l = mom_idx%beg, mom_idx%end
                                            DEALLOCATE(qL_prim_ndqp(i,j,k)%vf(l)%sf)
                                            DEALLOCATE(qR_prim_ndqp(i,j,k)%vf(l)%sf)
                                        END DO
                                        IF (model_eqns == 3) THEN
                                            DO l = internalEnergies_idx%beg, internalEnergies_idx%end
                                                DEALLOCATE(qL_prim_ndqp(i,j,k)%vf(l)%sf)
                                                DEALLOCATE(qR_prim_ndqp(i,j,k)%vf(l)%sf)
                                            END DO
                                        END IF
                                    END IF
                                END IF
                               
                                DO l = 1, crv_size
                                    DEALLOCATE(qL_cons_ndqp(i,j,k)%vf( &
                                                 E_idx+crv_idx(l))%sf  )
                                    DEALLOCATE(qR_cons_ndqp(i,j,k)%vf( &
                                                 E_idx+crv_idx(l))%sf  )
                                END DO
                               
                                DO l = 1, sys_size
                                    NULLIFY(qL_cons_ndqp(i,j,k)%vf(l)%sf)
                                    NULLIFY(qR_cons_ndqp(i,j,k)%vf(l)%sf)
                                    NULLIFY(qL_prim_ndqp(i,j,k)%vf(l)%sf)
                                    NULLIFY(qR_prim_ndqp(i,j,k)%vf(l)%sf)
                                END DO
                               
                            ELSE
                               
                                DO l = 1, cont_idx%end
                                    DEALLOCATE(qL_cons_ndqp(i,j,k)%vf(l)%sf)
                                    DEALLOCATE(qR_cons_ndqp(i,j,k)%vf(l)%sf)
                                END DO
                               
                                IF(weno_vars == 1) THEN
                                    DO l = mom_idx%beg, E_idx
                                        DEALLOCATE(qL_cons_ndqp(i,j,k)%vf(l)%sf)
                                        DEALLOCATE(qR_cons_ndqp(i,j,k)%vf(l)%sf)
                                    END DO
                                END IF
                               
                                DO l = mom_idx%beg, E_idx
                                    DEALLOCATE(qL_prim_ndqp(i,j,k)%vf(l)%sf)
                                    DEALLOCATE(qR_prim_ndqp(i,j,k)%vf(l)%sf)
                                END DO
                                IF (model_eqns == 3) THEN
                                    DO l = internalEnergies_idx%beg, internalEnergies_idx%end
                                        DEALLOCATE(qL_prim_ndqp(i,j,k)%vf(l)%sf)
                                        DEALLOCATE(qR_prim_ndqp(i,j,k)%vf(l)%sf)
                                    END DO
                                END IF
                               
                                DO l = adv_idx%beg, adv_idx%end
                                    DEALLOCATE(qL_cons_ndqp(i,j,k)%vf(l)%sf)
                                    DEALLOCATE(qR_cons_ndqp(i,j,k)%vf(l)%sf)
                                END DO
                               
                            END IF
                        
                        END IF
                     
                        DEALLOCATE(qL_cons_ndqp(i,j,k)%vf, qL_prim_ndqp(i,j,k)%vf)
                        DEALLOCATE(qR_cons_ndqp(i,j,k)%vf, qR_prim_ndqp(i,j,k)%vf)
                     
                    END DO
                END DO
            END DO
            
            DEALLOCATE(qL_cons_ndqp, qR_cons_ndqp, qL_prim_ndqp, qR_prim_ndqp)
            ! END: Deallocation/Disassociation of qK_cons_ndqp and qK_prim_ndqp
            
            
            ! Deallocation of dq_prim_ds_qp ====================================
            IF(ANY(Re_size > 0) .OR. We_size > 0) THEN
               
                DO k = itau%beg, itau%end
                    DO j = iksi%beg, iksi%end
                        DO i = ieta%beg, ieta%end
                        
                            IF(ALL((/i,j,k/) == 0) .AND. ANY(Re_size > 0)) THEN
                           
                                DO l = mom_idx%beg, mom_idx%end
                                    DEALLOCATE(dq_prim_dx_qp(i,j,k)%vf(l)%sf)
                                    DEALLOCATE(gm_vel_qp(i,j,k)%vf(l)%sf)
                                END DO
                           
                                IF(n > 0) THEN
                              
                                    DO l = mom_idx%beg, mom_idx%end
                                        DEALLOCATE(dq_prim_dy_qp(i,j,k)%vf(l)%sf)
                                    END DO
                              
                                    IF(p > 0) THEN
                                        DO l = mom_idx%beg, mom_idx%end
                                            DEALLOCATE(dq_prim_dz_qp(i,j,k)%vf(l)%sf)
                                        END DO
                                    END IF
                              
                                END IF
                           
                            END IF
                        
                            IF( (i /= 0 .AND. ABS(j) >= ABS(k)) &
                                              .OR.              &
                                  ALL((/i,j,k/) == 0)       ) THEN
                                DO l = 1, crv_size
                                    DEALLOCATE(dq_prim_dx_qp(i,j,k)%vf( &
                                                    E_idx+crv_idx(l))%sf)
                                    IF (n > 0) THEN
                                        DEALLOCATE(dq_prim_dy_qp(i,j,k)%vf( &
                                                        E_idx+crv_idx(l))%sf)
                                        IF (p > 0) THEN
                                            DEALLOCATE(dq_prim_dz_qp(i,j,k)%vf( &
                                                            E_idx+crv_idx(l))%sf)
                                        END IF
                                    END IF
                                END DO
                            END IF
                        
                            DEALLOCATE(dq_prim_dx_qp(i,j,k)%vf)
                            DEALLOCATE(dq_prim_dy_qp(i,j,k)%vf)
                            DEALLOCATE(dq_prim_dz_qp(i,j,k)%vf)
                            DEALLOCATE(gm_vel_qp(i,j,k)%vf)
                        
                        END DO
                    END DO
                END DO
               
                DEALLOCATE(dq_prim_dx_qp, dq_prim_dy_qp, dq_prim_dz_qp, gm_vel_qp)
               
            END IF
            ! END: Deallocation of dq_prim_ds_qp ===============================
            
            
            ! Deallocation/Disassociation of dqK_prim_ds_ndqp ==================
            IF(ANY(Re_size > 0) .OR. We_size > 0) THEN
                DO k = ipsi%beg, ipsi%end
                    DO j = ichi%beg, ichi%end
                        DO i = num_dims, 1, -1
                        
                            IF(ABS(j) >= ABS(k)) THEN
                           
                                IF(ANY(Re_size > 0)) THEN
                              
                                    DO l = mom_idx%beg, mom_idx%end
                                        DEALLOCATE(dqL_prim_dx_ndqp(i,j,k)%vf(l)%sf)
                                        DEALLOCATE(dqR_prim_dx_ndqp(i,j,k)%vf(l)%sf)
                                    END DO
                              
                                    IF(n > 0) THEN
                                        DO l = mom_idx%beg, mom_idx%end
                                            DEALLOCATE(dqL_prim_dy_ndqp(i,j,k)%vf(l)%sf)
                                            DEALLOCATE(dqR_prim_dy_ndqp(i,j,k)%vf(l)%sf)
                                        END DO
                                    END IF
                              
                                    IF(p > 0) THEN
                                        DO l = mom_idx%beg, mom_idx%end
                                            DEALLOCATE(dqL_prim_dz_ndqp(i,j,k)%vf(l)%sf)
                                            DEALLOCATE(dqR_prim_dz_ndqp(i,j,k)%vf(l)%sf)
                                        END DO
                                    END IF
                              
                                END IF
                           
                                DO l = 1, crv_size
                                    DEALLOCATE(dqL_prim_dx_ndqp(i,j,k)%vf( &
                                                     E_idx+crv_idx(l))%sf  )
                                    DEALLOCATE(dqR_prim_dx_ndqp(i,j,k)%vf( &
                                                     E_idx+crv_idx(l))%sf  )
                                    IF (n > 0) THEN
                                        DEALLOCATE(dqL_prim_dy_ndqp(i,j,k)%vf( &
                                                         E_idx+crv_idx(l))%sf  )
                                        DEALLOCATE(dqR_prim_dy_ndqp(i,j,k)%vf( &
                                                         E_idx+crv_idx(l))%sf  )
                                    END IF
                                END DO
                           
                                IF(p > 0) THEN
                                    DO l = 1, crv_size
                                        DEALLOCATE(dqL_prim_dz_ndqp(i,j,k)%vf( &
                                                         E_idx+crv_idx(l))%sf  )
                                        DEALLOCATE(dqR_prim_dz_ndqp(i,j,k)%vf( &
                                                         E_idx+crv_idx(l))%sf  )
                                    END DO
                                END IF
                           
                            END IF
                        
                            DEALLOCATE(dqL_prim_dx_ndqp(i,j,k)%vf)
                            DEALLOCATE(dqL_prim_dy_ndqp(i,j,k)%vf)
                            DEALLOCATE(dqL_prim_dz_ndqp(i,j,k)%vf)
                            DEALLOCATE(dqR_prim_dx_ndqp(i,j,k)%vf)
                            DEALLOCATE(dqR_prim_dy_ndqp(i,j,k)%vf)
                            DEALLOCATE(dqR_prim_dz_ndqp(i,j,k)%vf)
                        
                        END DO
                    END DO
                END DO
            END IF
            
            DEALLOCATE(dqL_prim_dx_ndqp, dqL_prim_dy_ndqp, dqL_prim_dz_ndqp)
            DEALLOCATE(dqR_prim_dx_ndqp, dqR_prim_dy_ndqp, dqR_prim_dz_ndqp)
            ! END: Deallocation/Disassociation of dqK_prim_ds_ndqp =============
            
            
            ! Deallocation of gm_alpha_qp ======================================
            IF(We_size > 0) THEN
                DO k = itau%beg, itau%end
                    DO j = iksi%beg, iksi%end
                        DO i = ieta%beg, ieta%end
                        
                            IF(ANY(SUM(ABS((/i,j,k/))) == (/0,num_dims/))) THEN
                                DO l = 1, crv_size
                                    DEALLOCATE(gm_alpha_qp(i,j,k)%vf(crv_idx(l))%sf)
                                END DO
                            END IF
                        
                            DEALLOCATE(gm_alpha_qp(i,j,k)%vf)
                        
                        END DO
                    END DO
                END DO
            END IF
            
            DEALLOCATE(gm_alpha_qp)
            ! ==================================================================
            
            
            ! Deallocation of gm_alphaK_ndqp ===================================
            IF(We_size > 0) THEN
                DO k = ipsi%beg, ipsi%end
                    DO j = ichi%beg, ichi%end
                        DO i = 1, num_dims
                        
                            IF(ANY(SUM(ABS((/j,k/))) == (/0,num_dims/))) THEN
                                DO l = 1, crv_size
                                    DEALLOCATE(gm_alphaL_ndqp(i,j,k)%vf( &
                                                         crv_idx(l))%sf  )
                                    DEALLOCATE(gm_alphaR_ndqp(i,j,k)%vf( &
                                                         crv_idx(l))%sf  )
                                END DO
                            END IF
                        
                            DEALLOCATE(gm_alphaL_ndqp(i,j,k)%vf)
                            DEALLOCATE(gm_alphaR_ndqp(i,j,k)%vf)
                        
                        END DO
                    END DO
                END DO
            END IF
            
            DEALLOCATE(gm_alphaL_ndqp, gm_alphaR_ndqp)
            ! ==================================================================
            
            
            ! Deallocation of un_alphaK_s_ndqp =================================
            IF(We_size > 0) THEN
               
                DO k = ipsi%beg, ipsi%end
                    DO j = ichi%beg, ichi%end
                     
                        IF(ABS(j) >= ABS(k)) THEN
                        
                            DO l = 1, crv_size
                                DEALLOCATE(un_alphaL_x_ndqp(1,j,k)%vf(crv_idx(l))%sf)
                                DEALLOCATE(un_alphaR_x_ndqp(1,j,k)%vf(crv_idx(l))%sf)
                                IF (n > 0) THEN
                                    DEALLOCATE(un_alphaL_y_ndqp(2,j,k)%vf(crv_idx(l))%sf)
                                    DEALLOCATE(un_alphaR_y_ndqp(2,j,k)%vf(crv_idx(l))%sf)
                                END IF
                            END DO
                        
                            IF(p > 0) THEN
                                DO l = 1, crv_size
                                    DEALLOCATE(un_alphaL_z_ndqp(3,j,k)%vf( &
                                                           crv_idx(l))%sf  )
                                    DEALLOCATE(un_alphaR_z_ndqp(3,j,k)%vf( &
                                                           crv_idx(l))%sf  )
                                END DO
                            END IF
                        
                        END IF
                     
                        DO i = 1, num_dims
                            DEALLOCATE(un_alphaL_x_ndqp(i,j,k)%vf)
                            DEALLOCATE(un_alphaL_y_ndqp(i,j,k)%vf)
                            DEALLOCATE(un_alphaL_z_ndqp(i,j,k)%vf)
                            DEALLOCATE(un_alphaR_x_ndqp(i,j,k)%vf)
                            DEALLOCATE(un_alphaR_y_ndqp(i,j,k)%vf)
                            DEALLOCATE(un_alphaR_z_ndqp(i,j,k)%vf)
                        END DO
                     
                    END DO
                END DO
               
                DEALLOCATE(un_alphaL_x_ndqp, un_alphaL_y_ndqp, un_alphaL_z_ndqp)
                DEALLOCATE(un_alphaR_x_ndqp, un_alphaR_y_ndqp, un_alphaR_z_ndqp)
               
            END IF
            ! ==================================================================
            
            
            ! Deallocation of kappa_vf =========================================
            IF(We_size > 0) THEN
               
                DO i = 1, crv_size
                    DEALLOCATE(kappa_vf(crv_idx(i))%sf)
                END DO
               
                DEALLOCATE(kappa_vf)
               
            END IF
            ! ==================================================================
            
            
            ! Deallocation/Disassociation of kappaK_ndqp =======================
            IF(We_size > 0) THEN
                DO k = ipsi%beg, ipsi%end
                    DO j = ichi%beg, ichi%end
                        DO i = num_dims, 1, -1
                        
                            IF(ABS(j) >= ABS(k)) THEN
                                IF(i /= 1) THEN
                                    DO l = 1, crv_size
                                        NULLIFY(kappaL_ndqp(i,j,k)%vf(crv_idx(l))%sf)
                                        NULLIFY(kappaR_ndqp(i,j,k)%vf(crv_idx(l))%sf)
                                    END DO
                                ELSE
                                    DO l = 1, crv_size
                                        DEALLOCATE(kappaL_ndqp(i,j,k)%vf( &
                                                          crv_idx(l))%sf  )
                                        DEALLOCATE(kappaR_ndqp(i,j,k)%vf( &
                                                          crv_idx(l))%sf  )
                                    END DO
                                END IF
                            END IF
                        
                            DEALLOCATE(kappaL_ndqp(i,j,k)%vf, kappaR_ndqp(i,j,k)%vf)
                        
                        END DO
                    END DO
                END DO
            END IF
            
            DEALLOCATE(kappaL_ndqp, kappaR_ndqp)
            ! ==================================================================
            
            IF (alt_crv) THEN
                ! Deallocation of dgm_alpha_ds_qp & un_alpha_s_qp
                IF(We_size > 0) THEN
                    DO k = itau%beg, itau%end
                        DO j = iksi%beg, iksi%end
                            DO i = ieta%beg, ieta%end
    
                                IF(ANY(SUM(ABS((/i,j,k/))) == (/0,num_dims/))) THEN
                                    DO l = 1, crv_size
                                        DEALLOCATE(dgm_alpha_dx_qp(i,j,k)%vf(crv_idx(l))%sf)
                                        DEALLOCATE(dgm_alpha_dy_qp(i,j,k)%vf(crv_idx(l))%sf)
                                        DEALLOCATE(dgm_alpha_dz_qp(i,j,k)%vf(crv_idx(l))%sf)
                                        DEALLOCATE(un_alpha_x_qp(i,j,k)%vf(crv_idx(l))%sf)
                                        DEALLOCATE(un_alpha_y_qp(i,j,k)%vf(crv_idx(l))%sf)
                                        DEALLOCATE(un_alpha_z_qp(i,j,k)%vf(crv_idx(l))%sf)
                                    END DO
                                END IF
    
                                DEALLOCATE(dgm_alpha_dx_qp(i,j,k)%vf)
                                DEALLOCATE(dgm_alpha_dy_qp(i,j,k)%vf)
                                DEALLOCATE(dgm_alpha_dz_qp(i,j,k)%vf)
                                DEALLOCATE(un_alpha_x_qp(i,j,k)%vf)
                                DEALLOCATE(un_alpha_y_qp(i,j,k)%vf)
                                DEALLOCATE(un_alpha_z_qp(i,j,k)%vf)
    
                            END DO
                        END DO
                    END DO
                END IF
    
                DEALLOCATE(dgm_alpha_dx_qp)
                DEALLOCATE(dgm_alpha_dy_qp)
                DEALLOCATE(dgm_alpha_dz_qp)
                DEALLOCATE(un_alpha_x_qp)
                DEALLOCATE(un_alpha_y_qp)
                DEALLOCATE(un_alpha_z_qp)
            END IF

            ! Deallocation of alt_kappa_vf & laplacian_vf
            IF(We_size > 0) THEN
                IF (alt_crv) THEN
                    DO i = 1, crv_size
                        DEALLOCATE(alt_kappa_vf(crv_idx(i))%sf)
                        DEALLOCATE(laplacian_vf(crv_idx(i))%sf)
                    END DO
            
                    DEALLOCATE(alt_kappa_vf)
                    DEALLOCATE(laplacian_vf)
                END IF

                IF (We_src) THEN
                    DEALLOCATE(We_mtm_src)
                    DEALLOCATE(We_nrg_src)
                END IF

            END IF
                
            IF (We_size > 0 .AND. We_rhs_flux) THEN
                DO i = 1, num_dims
                    DEALLOCATE(tau_We_vf(cont_idx%end+i)%sf)
                END DO
                DEALLOCATE(tau_We_vf(E_idx)%sf)
                                            
                DEALLOCATE(tau_We_vf)
            END IF    

            IF (ANY(Re_size > 0) .AND. cyl_coord) THEN
                DO i = 1, num_dims
                    DEALLOCATE(tau_Re_vf(cont_idx%end+i)%sf)
                END DO
                DEALLOCATE(tau_Re_vf(E_idx)%sf)
                                            
                DEALLOCATE(tau_Re_vf)
            END IF

            ! Deallocation of reg_src_vf
            IF (regularization) THEN
                DO i = 1, sys_size
                    DEALLOCATE(reg_src_vf(i)%sf)
                END DO
                DEALLOCATE(reg_src_vf)
            END IF

            IF (tvd_rhs_flux) THEN
                ! Deallocation of flux limiter variables
                DEALLOCATE(flux_lim_func)

                DO k = ipsi%beg, ipsi%end
                    DO j = ichi%beg, ichi%end
                        DO i = num_dims, 1, -1
                            IF (ABS(j) >= ABS(k)) THEN
                                IF (i /= 1) THEN
                                    DO l = 1, sys_size
                                        NULLIFY(    lo_flux_ndqp(i,j,k)%vf(l)%sf)
                                        NULLIFY(    hi_flux_ndqp(i,j,k)%vf(l)%sf)
                                        NULLIFY(lo_flux_src_ndqp(i,j,k)%vf(l)%sf)
                                        NULLIFY(hi_flux_src_ndqp(i,j,k)%vf(l)%sf)
                                        NULLIFY(lo_flux_gsrc_ndqp(i,j,k)%vf(l)%sf)
                                        NULLIFY(hi_flux_gsrc_ndqp(i,j,k)%vf(l)%sf)
                                    END DO
                                ELSE
                                    DO l = 1, sys_size
                                        DEALLOCATE(lo_flux_ndqp(i,j,k)%vf(l)%sf)
                                        DEALLOCATE(hi_flux_ndqp(i,j,k)%vf(l)%sf)
                                        DEALLOCATE(lo_flux_gsrc_ndqp(i,j,k)%vf(l)%sf)
                                        DEALLOCATE(hi_flux_gsrc_ndqp(i,j,k)%vf(l)%sf)
                                    END DO

                                    IF (ANY(Re_size > 0) .OR. We_size > 0) THEN
                                        DO l = mom_idx%beg, E_idx
                                            DEALLOCATE(lo_flux_src_ndqp(i,j,k)%vf(l)%sf)
                                            DEALLOCATE(hi_flux_src_ndqp(i,j,k)%vf(l)%sf)
                                        END DO
                                    END IF
                                    
                                    IF (riemann_solver == 1) THEN
                                        DO l = adv_idx%beg+1, adv_idx%end
                                            DEALLOCATE(lo_flux_src_ndqp(i,j,k)%vf(l)%sf)
                                            DEALLOCATE(hi_flux_src_ndqp(i,j,k)%vf(l)%sf)
                                        END DO
                                    ELSE
                                        DO l = adv_idx%beg+1, adv_idx%end
                                            NULLIFY(lo_flux_src_ndqp(i,j,k)%vf(l)%sf)
                                            NULLIFY(hi_flux_src_ndqp(i,j,k)%vf(l)%sf)
                                        END DO
                                    END IF

                                    DEALLOCATE(lo_flux_src_ndqp(i,j,k)%vf(adv_idx%beg)%sf)
                                    DEALLOCATE(hi_flux_src_ndqp(i,j,k)%vf(adv_idx%beg)%sf)
                                END IF
                            END IF
                            
                            DEALLOCATE(lo_flux_ndqp(i,j,k)%vf, lo_flux_src_ndqp(i,j,k)%vf, lo_flux_gsrc_ndqp(i,j,k)%vf)
                            DEALLOCATE(hi_flux_ndqp(i,j,k)%vf, hi_flux_src_ndqp(i,j,k)%vf, hi_flux_gsrc_ndqp(i,j,k)%vf)
                        END DO
                    END DO
                END DO

                DEALLOCATE(lo_flux_ndqp, lo_flux_src_ndqp, lo_flux_gsrc_ndqp)
                DEALLOCATE(hi_flux_ndqp, hi_flux_src_ndqp, hi_flux_gsrc_ndqp)
            END IF

            ! Deallocation/Disassociation of flux_ndqp, flux_src_ndqp, and flux_gsrc_ndqp ====
            DO k = ipsi%beg, ipsi%end
                DO j = ichi%beg, ichi%end
                    DO i = num_dims, 1, -1
                     
                        IF(ABS(j) >= ABS(k)) THEN
                            IF(i /= 1) THEN
                           
                                DO l = 1, sys_size
                                    NULLIFY(    flux_ndqp(i,j,k)%vf(l)%sf)
                                    NULLIFY(flux_src_ndqp(i,j,k)%vf(l)%sf)
                                    NULLIFY(flux_gsrc_ndqp(i,j,k)%vf(l)%sf)
                                END DO
                           
                            ELSE
                           
                                DO l = 1, sys_size
                                    DEALLOCATE(flux_ndqp(i,j,k)%vf(l)%sf)
                                    DEALLOCATE(flux_gsrc_ndqp(i,j,k)%vf(l)%sf)
                                END DO
                           
                                IF(ANY(Re_size > 0) .OR. We_size > 0) THEN
                                    DO l = mom_idx%beg, E_idx
                                        DEALLOCATE(flux_src_ndqp(i,j,k)%vf(l)%sf)
                                    END DO
                                END IF
                           
                                IF(riemann_solver == 1) THEN
                                    DO l = adv_idx%beg+1, adv_idx%end
                                        DEALLOCATE(flux_src_ndqp(i,j,k)%vf(l)%sf)
                                    END DO
                                ELSE
                                    DO l = adv_idx%beg+1, adv_idx%end
                                        NULLIFY(flux_src_ndqp(i,j,k)%vf(l)%sf)
                                    END DO
                                END IF
                           
                                DEALLOCATE(flux_src_ndqp(i,j,k)%vf(adv_idx%beg)%sf)
                           
                            END IF
                        END IF
                     
                        DEALLOCATE(flux_ndqp(i,j,k)%vf, flux_src_ndqp(i,j,k)%vf, flux_gsrc_ndqp(i,j,k)%vf)
                     
                    END DO
                END DO
            END DO
            
            DEALLOCATE(flux_ndqp, flux_src_ndqp, flux_gsrc_ndqp)
                     
            ! END: Deallocation/Disassociation of flux_ndqp, flux_src_ndqp, and flux_gsrc_ndqp  ===
            
            
            ! Disassociating procedural pointer to the subroutine which was
            ! utilized to calculate the solution of a given Riemann problem
            s_riemann_solver => NULL()
            
            ! Disassociating the pointer to the procedure that was utilized to
            ! to convert mixture or species variables to the mixture variables
            s_convert_to_mixture_variables => NULL()
            
        END SUBROUTINE s_finalize_rhs_module ! ---------------------------------
        
END MODULE m_rhs