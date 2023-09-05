#include "common.h"


int QDWHpartial ( int M, int N, 
                        int fact, int psinv,
                        double s, 
                        double tol,
                        double *A, int iA, int jA, int *descA, 
                        double *S, 
                        double *U, int iU, int jU, int *descU, 
                        double *VT, int iVT, int jVT, int *descVT, 
                        double *B, int iB, int jB, int *descB, 
                        int *sizeS, int *sizeK,
                        int *it, 
                        double *flops)
{
    
    //#define d_A(i_,j_) (d_A + (i_) + (j_)*ldda1)

    int i, j, ii, info, flip;
    int nbm, lwork;
    int sizeQ1, sizeQ1_r32;
    int Mtmp = M, Ntmp = N;

    double *Q1, *Acpy, *T, *Qini, *tau;
    int descQ1[9], descAcpy[9], descT[9], descQini[9];

    double *Usub, *VTsub;
    int descUsub[9], descVTsub[9];

    double alpha, beta, scl;

    int lwork_qr;
    double *Work_qr1, *Work_qr2, *Work_svd;
    char *jobu = "V"; char *jobvt = "V";
    

    int i1 = 1, i0 = 0;
    int max_mn = max(M,N);
    int min_mn = min(M,N);

    Acpy = U;
    T    = VT;
    tau  = S;

   /* Quick return if possible */
    if ( M == 0 || N == 0 ) {
        return 0;
    }
    /* Check the inputs */
    if ( M < 0 || N < 0 ){
        printf("\n matrix with negative dimension \n");
        return 1;
    }

    /*
     * Scale the matrix s.t. ||A||_2=1
     * Scale by ||A||_f could be enough because ||A||_2 <= ||A||_f 
     */
    scl  = 1.0;//pdlange_ ( "f", &M, &N, A, &i1, &i1, descA, S);
    alpha = 1.0;
    pdlascl_( "G", &scl, &alpha, 
              &M, &N, 
              A, &i1, &i1, descA, 
              &info);


    /*
     * Get the grid parameters
     */
    int mloc, nloc, mlocW, nb;   
    int myrow, mycol, nprow, npcol;   
    int mloc_max_mn, nloc_max_mn;
    int mloc_min_mn, nloc_min_mn;
    int mloc_n, nloc_n;
    int mloc_pinv, nloc_pinv;
    int ctxt_ = 1, nb_ = 5;
    int ictxt;
    int MB = 2*min_mn; 

    ictxt = descU[ctxt_];
    Cblacs_get( -1, 0, &ictxt );
    nb = descU[nb_];
    Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );

    mloc  = numroc_( &M,  &nb, &myrow, &i0, &nprow );
    nloc  = numroc_( &N,  &nb, &mycol, &i0, &npcol );

    mloc_min_mn  = numroc_( &min_mn, &nb, &myrow, &i0, &nprow );
    nloc_min_mn  = numroc_( &min_mn, &nb, &mycol, &i0, &npcol );

    mloc_max_mn  = numroc_( &max_mn, &nb, &myrow, &i0, &nprow );
    nloc_max_mn  = numroc_( &max_mn, &nb, &mycol, &i0, &npcol );

    mlocW = numroc_( &MB, &nb, &myrow, &i0, &nprow );

    mloc_pinv     = numroc_( &N, &nb, &myrow, &i0, &nprow );
    nloc_pinv     = numroc_( &M, &nb, &mycol, &i0, &npcol );

    descinit_( descAcpy, &M, &N, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
    /*
     * Initial QR to reduce to square case NxN
     */
    flip = 0;
    if ( M != N )
    { 
        /*
         * Allocate for the initial QR to reduce to square case min(M,N)xmin(M,N)
         */
        if ( M < N ){

            Qini  = (double *)malloc(mloc_max_mn*nloc_max_mn*sizeof(double)) ;
            descinit_( descQini, &N, &M, &nb, &nb, &i0, &i0, &ictxt, &mloc_max_mn, &info );
            /*
             * Flip for fat matrix.  
             */
            alpha = 1.0; beta = 0.0;
            pdgeadd_( "T", &M, &N, 
                      &alpha, A,    &i1, &i1, descA, 
                      &beta,  Qini, &i1, &i1, descQini );
            M = Ntmp; N = Mtmp;
            flip = 1;
        }
        else{
            Qini  = (double *)malloc(mloc*nloc*sizeof(double)) ;
            descinit_( descQini, &M, &N, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
            /*
             * Copy the matrix.  
             */
            pdlacpy_( "A", &M, &N, 
                      A,    &i1, &i1, descA, 
                      Qini, &i1, &i1, descQini );

        }
        //N32 = r32up(N);

        /*
         * Initial QR to reduce to square case NxN
         */
        //nbm = magma_get_sgeqrf_nb( M, N );  
        int lwork_qr = -1;
        Work_qr1 = (double *)calloc(1,sizeof(double)) ;
        pdgeqrf_( &M, &N, 
                 Qini, &i1, &i1, descQini, 
                 tau, 
                 Work_qr1, &lwork_qr, 
                 &info );
        lwork_qr  = Work_qr1[0];
        Work_qr1 = (double *)calloc(lwork_qr,sizeof(double)) ;

        pdgeqrf_( &M, &N, 
                 Qini, &i1, &i1, descQini, 
                 tau, 
                 Work_qr1, &lwork_qr, 
                 &info );

        /*
         * Build the upper triangular factor R (d_A) to find its SVD
         * Copy diagonal blocks from d_T back to d_A
         * Zeroing out below it
         */
        pdlacpy_( "A", &M, &N, 
                   Qini, &i1, &i1, descQini, 
                   A,    &i1, &i1, descA );

        int N_1 = N - 1;
        alpha = 0.0;  
        //pdlaset_( "G", &N_1, &N_1, 
        //          &alpha, &alpha, 
        //          A+1, &i1, &i1, descA);

        /*
         * Build Q factor (d_Qini) to accumulate into the left singular vectors
         */
        pdorgqr_( &M, &N, &N, 
                  Qini, &i1, &i1, descQini, 
                  tau, 
                  Work_qr1, &lwork_qr, 
                  &info );

        /*
         * Backup d_A in d_Acpy, the copy is needed to find the svd(d_A * d_Q1) 
         */
        pdlacpy_( "A", &N, &N, 
                   A,    &i1, &i1, descA, 
                   Acpy, &i1, &i1, descAcpy );

        /* Main flops used in this step */
        *flops += FLOPS_SGEQRF( M, N);
        *flops += FLOPS_SORGQR( M, N, N);
    }

    else{
        /*
         * Backup d_A in d_Acpy, the copy is needed to find the svd(d_A * d_Q1) 
         */
        pdlacpy_( "A", &N, &N, 
                   A,    &i1, &i1, descA, 
                   Acpy, &i1, &i1, descAcpy );
    }

    descinit_( descT, &N, &N, &nb, &nb, &i0, &i0, &ictxt, &mloc_min_mn, &info );
    /* 
     * Now we call QDWH on square matrix d_Acpy with size min(M,N)xmin(M,N)
     * s is an input. 
     * U_p = d_Acpy = QDWH(d_Acpy): This computes the orthogonal polar factor d_Acpy with the property that 
     * (i) the singular vectors are the same as d_Acpy; 
     * (ii) the singular values of d_Acpy in [s,1] are mapped to 1 to within machine precision. 
     */

    int lWork1, lWork2;
    double *Wloc1, *Wloc2;
    Wloc1  = (double *)calloc(1,sizeof(double)) ;
    Wloc2  = (double *)calloc(1,sizeof(double)) ;
    lWork1 = -1; 
    lWork2 = -1; 

    pdgeqdwh( "N", N, N, 
              fact, 
              s,
              Acpy, i1, i1, descAcpy, 
              VT, i1, i1, descVT, 
              Wloc1, lWork1,
              Wloc2, lWork2,
              it, 
              flops,
              &info);

    lWork1 = Wloc1[0];
    lWork2 = Wloc2[0];

        
    //Wloc  = (double *)malloc(lWork*n*sizeof(double));
    Wloc1  = (double *)malloc((lWork1*nloc_min_mn)*sizeof(double));
    Wloc2  = (double *)malloc((lWork2*nloc_min_mn)*sizeof(double));

    pdgeqdwh( "N", N, N, 
              fact, 
              s,
              Acpy, i1, i1, descAcpy, 
              VT, i1, i1, descVT, 
              Wloc1, lWork1,
              Wloc2, lWork2,
              it, 
              flops,
              &info);


    /*
     * [Q,R] = qr(eye(length(U))-U_p'*U_p), where U_p = d_Acpy
     * To get the desired eigenspace we form the (full) QR factorization of $U_p+I$, and take
     * the 'null space'. Here we need to be careful as eigenvectors are ill-conditioned if the
     * gap is small. This is controlled by the choice of tol: the smaller tol,
     * the more efficient (smaller Q1), but worse final accuracy. 
     */
    alpha = 0.0; beta =1.0; 
    pdlaset_( "G", &N, &N, 
              &alpha, &beta, 
              T, &i1, &i1, descT );
    alpha = -1.0; beta = 1.0;
    pdgemm_( "T", "N", 
             &N, &N, &N, 
             &alpha, Acpy, &i1, &i1, descAcpy, 
                     Acpy, &i1, &i1, descAcpy, 
             &beta,  T,    &i1, &i1, descT );

    lwork_qr = -1;
    Work_qr2 = (double *)calloc(1,sizeof(double)) ;
    pdgeqrf_( &N, &N, 
              T, &i1, &i1, descT, 
              tau, 
              Work_qr2, &lwork_qr, 
              &info );
    lwork_qr = Work_qr2[0];
    Work_qr2 = (double *)calloc(lwork_qr,sizeof(double)) ;

    pdgeqrf_( &N, &N, 
              T, &i1, &i1, descT, 
              tau, 
              Work_qr2, &lwork_qr, 
              &info );

    /*
     * ii = min(find(abs(diag(R))<tol));
     * Build the upper triangular factor R (d_Acpy) to find the projected size
     */
    ii = N;
    for ( i = 1; i <= N; i++ ) {
            int idum1, idum2, iloc, jloc;
            if ( ( myrow == indxg2p_( &i, &nb, &idum1, &i0, &nprow ) )
            &&   ( mycol == indxg2p_( &i, &nb, &idum1, &i0, &npcol ) ) ){
                    iloc = indxg2l_( &i, &nb, &idum1, &idum2, &nprow );
                    jloc = indxg2l_( &i, &nb, &idum1, &idum2, &npcol );
                    alpha = T[ (jloc-1)*mloc_min_mn + (iloc-1) ];
                    if ( fabs(alpha) < tol ){
                       ii = i;
                       i = N + 1;
                    }
            }
    } 

    int ii_min;
    MPI_Allreduce( &ii, &ii_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if ( ii_min == N ){
        ii_min = 0;
    }
    ii = ii_min;

    /*
     * Build the Q factor to compute d_A*d_Q1 and find its SVD
     * d_Q1 is the first N-ii columns of the Q factor
     * d_Q1 is supposed to contain the desired eigenspace and a bit more
     * The Rayleigh-Ritz process is ideal for extracting extremal eigenpairs
     */
    pdorgqr_( &N, &N, &N, 
              T, &i1, &i1, descT, 
              tau, 
              Work_qr2, &lwork_qr, 
              &info );

    sizeQ1 = N - ii;
    //Q1 = &T[ii*N];

    /* Main flops used in this step */
    *flops += FLOPS_SGEMM(N, N, N);
    *flops += FLOPS_SGEQRF( N, N);
    *flops += FLOPS_SORGQR( N, N, N);

    //sizeQ1_r32 = r32up(sizeQ1);
    *sizeS = sizeQ1;
    
    int mloc_Usub, nloc_Usub; 
    int mloc_VTsub, nloc_VTsub; 

    mloc_Usub  = numroc_( &N, &nb, &myrow, &i0, &nprow );
    nloc_Usub  = numroc_( &sizeQ1, &nb, &mycol, &i0, &npcol );

    mloc_VTsub  = numroc_( &sizeQ1, &nb, &myrow, &i0, &nprow );
    nloc_VTsub  = numroc_( &sizeQ1, &nb, &mycol, &i0, &npcol );

    /*
     * Compute d_B = d_A*d_Q1 (A*Q1) with size (min(M,N)xsizeQ1)
     * [UU,SS,VV] = svd(d_A*d_Q1,0);
     */
    Usub  = (double *)malloc(mloc_Usub*nloc_Usub*sizeof(double)) ;
    descinit_( descUsub, &N, &sizeQ1, &nb, &nb, &i0, &i0, &ictxt, &mloc_Usub, &info );
//    VTsub = (double *)malloc(mloc_VTsub*nloc_VTsub*sizeof(double)) ;
//    descinit_( descVTsub, &sizeQ1, &sizeQ1, &nb, &nb, &i0, &i0, &ictxt, &mloc_VTsub, &info );

    int ii_ = ii + 1;
    alpha = 1.0; beta = 0.0;
    pdgemm_( "N", "N", 
             &N, &sizeQ1, &N, 
             &alpha, A,  &i1, &i1,  descA, 
                     T, &i1, &ii_, descT, 
             &beta,  B,  &i1, &i1, descB );
    *flops += FLOPS_SGEMM( N, sizeQ1, N);

    /*
     * Call GESDD to find SVD of d_B = d_A*d_Q1 (AxQ1)
     */ 

    /*
     * Work space required by GESDD 
     */ 
    lwork = -1;
    Work_svd = (double *)malloc(1*sizeof(double));
    pdgesvd_( jobu, jobvt, 
              &N, &sizeQ1, 
              B, &i1, &i1, descB, 
              S, 
              U,  &i1, &i1, descU, 
              VT, &i1, &i1, descVT, 
              Work_svd, &lwork, 
              &info );
    
    lwork = Work_svd[0];
    Work_svd = (double *)malloc(lwork*sizeof(double));
    /* 
     * U  (Nx     sizeQ1) is the matrix of the left singular vectors
     * S  (sizeQ1xsizeQ1) is the matrix of the singular values
     * VT (sizeQ1xsizeQ1) is the matrix of the right singular vectors
     */
    pdgesvd_( jobu, jobvt, 
              &N, &sizeQ1, 
              B, &i1, &i1, descB, 
              S, 
              U,  &i1, &i1, descU, 
              VT, &i1, &i1, descVT, 
              Work_svd, &lwork, 
              &info );
    
    if ( N > sizeQ1 ){
        int min_Ns = min(N,sizeQ1);
        int max_Ns = max(N,sizeQ1);
             // = (2*max_mn*min_mn^2      - 2/3*min_mn^3)         + (22*min_mn^3)          + 2*min_mn^3
        *flops += 2.*(double)max_Ns*(double)min_Ns*(double)min_Ns - 
                  2./3.*(double)min_Ns*(double)min_Ns*(double)min_Ns + 
                  22.*(double)min_Ns*(double)min_Ns*(double)min_Ns + 
                  2.*(double)min_Ns*(double)min_Ns*(double)min_Ns;
    }
    else{
        *flops += 22.*(double)N*(double)N*(double)N;
    }

    /* 
     * Set the singular values on the device
     */
    alpha = 1.0; beta = 0.0;

    /*
     * Accumulate the right singular vectors d_VT = d_Q1*d_B (V0 = Q1*VV)
     * d_VT = d_Q1*d_B  : d_Q1   (N     xsizeQ1) 
     *                  : d_B    (sizeQ1xsizeQ1) 
     *                  : d_B[N] (N     xsizeQ1)
     * d_B[N] ===> d_VT
     */
    pdgemm_( "N", "T", 
             &N, &sizeQ1, &sizeQ1, 
             &alpha, T,     &i1, &ii_, descT, 
                     //VTsub,  &i1, &i1, descVTsub, 
                     VT,  &i1, &i1, descVT, 
             &beta,  B,      &i1, &i1, descB );

    *flops += FLOPS_SGEMM( N, sizeQ1, sizeQ1);

    pdlacpy_( "A", &N, &sizeQ1, 
              B,  &i1, &i1, descB, 
              VT, &i1, &i1, descVT ); 

    /*
     * k: The number of the wanted singular values based on the Threshold (s): (sigma >= s)
     * Multiply the computed singular values by ||A||_f
     */
    int k = 0;
    for ( i = 0; i < sizeQ1; i++ ){
        S[i] = scl * S[i];
        if ( S[i] >= s ){
           k++;
        }
    }
    *sizeK = k;

    /*
     * Set the wanted singular values on the diagonal
     */
    if ( psinv ){
        alpha = 0.0;
        pdlaset_( "G", &k, &k, 
                  &alpha, &alpha, 
                  B, &i1, &i1, descB );

        for ( i = 0; i <= k; i++ ) {
               int idum1, idum2, iloc, jloc;
                if ( ( myrow == indxg2p_( &i, &nb, &idum1, &i0, &nprow ) )
                &&   ( mycol == indxg2p_( &i, &nb, &idum1, &i0, &npcol ) ) ){
                       iloc = indxg2l_( &i, &nb, &idum1, &idum2, &nprow );
                       jloc = indxg2l_( &i, &nb, &idum1, &idum2, &npcol );
                       B[ (jloc-1)*mloc + (iloc-1) ] = 1.0/S[i];
               }
        } 
    }

    alpha = 1.0; beta = 0.0;
    if ( M > N )
    {
        /*
         * When initial QR used for m>1.15n.
         * Accumulate the left singular vectors d_A = d_Qini*d_U 
         */
        pdgemm_( "N", "N", 
                 &M, &sizeQ1, &N, 
                 &alpha, Qini, &i1, &i1, descQini, 
                         Usub, &i1, &i1, descUsub, 
                 &beta,  A,    &i1, &i1, descA );

        /*
         * Compute the psuedo inverse (pinv) of non-square matrix A (MxN, M > N)
         * The pseudo inverse is d_VT*Sigma^-1*U^T 
         * d_U (Mxk)  is in d_A
         */
        if ( psinv ){
           pdgemm_( "N", "N", 
                    &N, &k, &k, 
                    &alpha, VTsub, &i1, &i1, descVTsub, 
                            B,     &i1, &i1, descB, 
                    &beta,  U,     &i1, &i1, descU );
           pdgemm_( "N", "T", 
                    &N, &M, &k, 
                    &alpha, U,    &i1, &i1, descU, 
                            A,    &i1, &i1, descA, 
                    &beta,  Qini, &i1, &i1, descQini );
        }


        if ( flip ){
            /*
             * flip singular vectors d_VT ===> d_U, d_U ===> d_VT
             */ 
            pdlacpy_( "A", &N, &sizeQ1, 
                      VT, &i1, &i1, descVT, 
                      U,  &i1, &i1, descU ); 
            pdlacpy_( "A", &M, &sizeQ1, 
                      A,  &i1, &i1, descA, 
                      VT, &i1, &i1, descVT ); 
            if ( psinv ){
               /* Transpose pinv if A is NxM, M > N */
               alpha = 1.0; beta = 0.0;
               pdgeadd_( "T", &N, &M, 
                         &alpha, Qini, &i1, &i1, descQini, 
                         &beta,  A,    &i1, &i1, descA );
               pdlacpy_( "A", &M, &N, 
                         A,    &i1, &i1, descA, 
                         Qini, &i1, &i1, descQini ); 
               /* Set the pinv on the CPU */
               //magma_ssetmatrix( M, N, 
               //                  d_Qini, ldda1, 
               //                  A,      lda, 
               //                  queue );
            }
        }
        else{
            /*
             * d_A ===> d_U
             */
            pdlacpy_( "A", &M, &sizeQ1, 
                      A, &i1, &i1, descA, 
                      U, &i1, &i1, descU ); 
            if ( psinv ){
               /* Set the pinv on the CPU */
               pdlacpy_( "A", &N, &M, 
                         Qini, &i1, &i1, descQini, 
                         A,    &i1, &i1, descA ); 
            }
        }

        *flops += FLOPS_SGEMM(M, sizeQ1, N);
    }
    else{  
            /*
             * Compute the psuedo inverse (pinv) of square matrix A
             */
            if ( psinv ){
               pdgemm_( "N", "N", 
                        &N, &k, &k, 
                        &alpha, VT, &i1, &i1, descVT, 
                                B,  &i1, &i1, descB, 
                        &beta,  A,  &i1, &i1, descA );
               pdgemm_( "N", "T", 
                        &N, &M, &k, 
                        &alpha, A, &i1, &i1, descA, 
                                U, &i1, &i1, descU, 
                        &beta,  B, &i1, &i1, descB );
               pdlacpy_( "A", &N, &M, 
                         B, &i1, &i1, descB, 
                         A, &i1, &i1, descA ); 
            }
    }
    
    if ( psinv ){
        *flops += FLOPS_SGEMM(N, k, k);
        *flops += FLOPS_SGEMM(N, M, k);
    }


    if ( M != N ){
       free(Qini);
       free(Work_qr1);
    }
    free(Work_qr2);
    free(Work_svd);
    free(Wloc1);
    free(Wloc2);
    return 0;
}
