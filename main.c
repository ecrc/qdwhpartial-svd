/**
 *
 * Copyright (c) 2017, King Abdullah University of Science and Technology
 * All rights reserved.
 *
 **/

/**
 *
 * @file testing_pdgeqdwh.c
 *
 *  QDWH is a high performance software framework for computing 
 *  the polar decomposition on distributed-memory manycore systems provided by KAUST
 *
 * @version 2.0.0
 * @author Dalal Sukkari
 * @author Hatem Ltaief
 * @date 2017-11-13
 *
 **/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <getopt.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>
#include <mkl_lapack.h>
#include <mkl_lapacke.h>
#include "myscalapack.h"
#include "include.h"
#include "flops.h"


/* Default values of parameters */
int nprow         = 1;
int npcol         = 1;
int lvec          = 1;
int rvec          = 1;
int m             = 5120;
int n             = 5120;
int nb            = 128;
int mode          = 0;
double cond       = 9.0072e+15;
int optcond       = 0;
int start         = 5120;
int stop          = 5120;
int step          = 1;
int niter         = 1;
int check         = 0;
int verbose       = 0;


void print_usage(void)
{
    fprintf(stderr,
            "======= QDWH testing using ScaLAPACK\n"
            " -p      --nprow         : Number of MPI process rows\n"
            " -q      --npcol         : Number of MPI process cols\n"
            " -jl     --lvec          : Compute left singular vectors\n"
            " -jr     --rvec          : Compute right singular vectors\n"
            " -m      --M             : First dimension of the matrix\n"
            " -n      --N             : Second dimension of the matrix\n"
            " -b      --nb            : Block size\n"
            " -d      --mode          : [1:6] Mode from pdlatms used to generate the matrix\n"
            " -k      --cond          : Condition number used to generate the matrix\n"
            " -o      --optcond       : Estimate Condition number using QR\n"
            " -i      --niter         : Number of iterations\n"
            " -r      --n_range       : Range for matrix sizes Start:Stop:Step\n"
            " -c      --check         : Check the solution\n"
            " -v      --verbose       : Verbose\n"
            " -h      --help          : Print this help\n" );
}

#define GETOPT_STRING "p:q:x:y:m:n:b:d:i:o:r:Q,S:s:w:e:c:f:t:v:h"

static struct option long_options[] =
    {
        /* PaRSEC specific options */
        {"nprow",      required_argument,  0, 'p'},
        {"npcol",      required_argument,  0, 'q'},
        {"jl",         no_argument,        0, 'x'},
        {"lvec",       no_argument,        0, 'x'},
        {"jr",         no_argument,        0, 'y'},
        {"rvec",       no_argument,        0, 'y'},
        {"M",          required_argument,  0, 'm'},
        {"m",          required_argument,  0, 'm'},
        {"N",          required_argument,  0, 'n'},
        {"n",          required_argument,  0, 'n'},
        {"nb",         required_argument,  0, 'b'},
        {"b",          required_argument,  0, 'b'},
        {"mode",       required_argument,  0, 'd'},
        {"d",          required_argument,  0, 'd'},
        {"cond",       required_argument,  0, 'k'},
        {"k",          required_argument,  0, 'k'},
        {"optcond",    required_argument,  0, 'o'},
        {"o",          required_argument,  0, 'o'},
        {"i",          required_argument,  0, 'i'},
        {"niter",      required_argument,  0, 'i'},
        {"r",          required_argument,  0, 'r'},
        {"n_range",    required_argument,  0, 'r'},
        {"check",      no_argument,        0, 'c'},
        {"verbose",    no_argument,        0, 'v'},
        {"help",       no_argument,        0, 'h'},
        {"h",          no_argument,        0, 'h'},
        {0, 0, 0, 0}
    };
static void parse_arguments(int argc, char** argv)
{
    int opt = 0;
    int c;
    int myrank_mpi;

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);

    do {
#if defined(HAVE_GETOPT_LONG)
        c = getopt_long_only(argc, argv, "",
                        long_options, &opt);
#else
        c = getopt(argc, argv, GETOPT_STRING);
        (void) opt;
#endif  /* defined(HAVE_GETOPT_LONG) */

        switch(c) {
        case 'p': nprow     = atoi(optarg); break;
        case 'q': npcol     = atoi(optarg); break;
        case 'm': n         = atoi(optarg); start = m; stop = m; step = 1; break;
        case 'n': n         = atoi(optarg); //start = n; stop = n; step = 1; break;
        case 'b': nb        = atoi(optarg); break;
        case 'd': mode      = atoi(optarg); break;
        case 'k': cond      = atof(optarg); break;
        case 'o': optcond   = atof(optarg); break;
        case 'i': niter     = atoi(optarg); break;
        case 'r': get_range( optarg, &start, &stop, &step ); break;
        case 'c': check     = 1; break;
        case 'v': verbose   = 1; break;
        case 'h':
            if (myrank_mpi == 0) print_usage(); MPI_Finalize(); exit(0);
            break;
        default:
            break;
        }
    } while(-1 != c);
}

int main(int argc, char **argv) {


    int myrank_mpi, nprocs_mpi;
    int ictxt, myrow, mycol;
    int mloc, nloc, mlocW;
    int mloc_min_mn, nloc_min_mn;
    int mloc_max_mn, nloc_max_mn;
    int mloc_n, nloc_n;
    int mloc_pinv, nloc_pinv;
    int mpi_comm_rows, mpi_comm_cols;
    int i, j, iter, size, info_facto_mr, info_facto_dc, info_facto_qw, info_facto_el, info_facto_sl, info, iseed;
    int my_info_facto;
    int i0 = 0, i1 = 1;
    int lwork, liwork, ldw, lwork_pdgesvd;


   /* Quick return if possible */
    if ( m == 0 || n == 0 ) {
        return 0;
    }
    /* Check the inputs */
    if ( m < 0 || n < 0 ){
        printf("\n matrix with negative dimension \n");
        return 1;
    }

    /* Allocation for the input/output matrices */
    //int descA[9], descH[9];
    //double *A=NULL, *H=NULL;

    /* Allocation to check the results */
    //int descAcpy[9], descC[9];
    //double *Acpy=NULL, *C=NULL;

    /* Allocation for pdlatsm */
    //double *Wloc1=NULL, *Wloc2=NULL, *D=NULL;

    double eps = LAPACKE_dlamch_work('e');
    int iprepad, ipostpad, sizemqrleft, sizemqrright, sizeqrf, sizeqtq,
        sizechk, sizesyevx, isizesyevx,
        sizesubtst, isizesubtst, sizetst,
        isizetst;
    double *D, *Wloc1;
/**/

    double flops, GFLOPS;

    double orth_Uqw, berr_UHqw;
    double frobA;

    double alpha, beta;
    char *jobu, *jobvt;


/**/



   if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "Program starts... \n");

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank_mpi);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs_mpi);

    if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "MPI Init done\n");

    parse_arguments(argc, argv);

    if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "Checking arguments done\n");

    Cblacs_get( -1, 0, &ictxt );
    Cblacs_gridinit( &ictxt, "R", nprow, npcol );
    Cblacs_gridinfo( ictxt, &nprow, &npcol, &myrow, &mycol );

    if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "BLACS Init done\n");

    if ( myrank_mpi == 0 ) {
        fprintf(stderr, "# \n");
        fprintf(stderr, "# NPROCS %d P %d Q %d\n", nprocs_mpi, nprow, npcol);
        fprintf(stderr, "# niter %d\n", niter);
        fprintf(stderr, "# n_range %d:%d:%d mode: %d cond: %2.4e \n", start, stop, step, mode, cond);
        fprintf(stderr, "# \n");
    }



    //int i, j;
    //int info;



    double my_elapsed_qwsvd, elapsed_qwsvd, sumtime_qwsvd, min_time_qwsvd, max_time_qwsvd;
    double my_elapsed_pdgesvd, elapsed_pdgesvd, sumtime_pdgesvd, min_time_pdgesvd, max_time_pdgesvd, flops_pdgesvd;

    long int min_mn, max_mn, n2;
    //int n232 = r32up(n2);
    //int lddb  = n232;
    
    //int ldd_min_mn = r32up(min_mn);
    //int ldd_max_mn = r32up(max_mn);
    //int ldda = ldd_max_mn;

    //int m32 = r32up(m);
    //int n32 = r32up(n);
    //int lddu  = m32;
    //int lddvt = n32;

    /*
     * Allocatation for PDGESVD 
     */
     double *Work_pdgesvd;

    /*
     * Allocatation for QDWHPartial
     */
    //float *d_A, *d_B, *d_VT, *d_U, *d_pinv, *S;  
    int descA[9], descB[9], descVT[9], descU[9], descpinv[9], descS[9];
    double *A=NULL, *B=NULL, *VT=NULL, *U=NULL, *pinv=NULL, *S=NULL;

    /*
     * Allocatation to check the results 
     */
    int descAcpy[9];
    double *Acpy=NULL, *W=NULL;

    int sizeS = 0, k = 0, ktmp, sizeStmp;

    if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "Range loop starts\n");

    // Begin loop over range
    for (size = start; size <= stop; size += step) {
        while ( (int)((double)size / (double)nb) < ( max(nprow , npcol) )){
            if ( myrank_mpi == 0 ) fprintf(stderr, " Matrix size is small to be facrorized using this number of processors \n");
               size += step;
        }
        m = size; 
        n = size;
        ldw = 2*min(m,n);
        min_mn = min(m,n);
        max_mn = max(m,n);
        n2 = 2*min_mn; 

        my_elapsed_qwsvd = 0.0, elapsed_qwsvd = 0.0, sumtime_qwsvd = 0.0, min_time_qwsvd = 0.0, max_time_qwsvd = 0.0;
        my_elapsed_pdgesvd = 0.0, elapsed_pdgesvd = 0.0, sumtime_pdgesvd = 0.0, min_time_pdgesvd = 0.0, max_time_pdgesvd = 0.0;

        mloc  = numroc_( &m, &nb, &myrow, &i0, &nprow );
        nloc  = numroc_( &n, &nb, &mycol, &i0, &npcol );

        mloc_min_mn  = numroc_( &min_mn, &nb, &myrow, &i0, &nprow );
        nloc_min_mn  = numroc_( &min_mn, &nb, &mycol, &i0, &npcol );

        mloc_max_mn  = numroc_( &max_mn, &nb, &myrow, &i0, &nprow );
        nloc_max_mn  = numroc_( &max_mn, &nb, &mycol, &i0, &npcol );

        mloc_n    = numroc_( &n, &nb, &myrow, &i0, &nprow );
        nloc_n    = numroc_( &n, &nb, &mycol, &i0, &npcol );

        mloc_pinv    = numroc_( &n, &nb, &myrow, &i0, &nprow );
        nloc_pinv    = numroc_( &m, &nb, &mycol, &i0, &npcol );

        mlocW = numroc_( &ldw, &nb, &myrow, &i0, &nprow );
//printf("\n mloc %d nloc %d mloc_min_mn %d nloc_min_mn %d mloc_max_mn %d nloc_max_mn %d mlocW %d mloc_pinv %d nloc_pinv %d \n", mloc, nloc, mloc_min_mn, nloc_min_mn, mloc_max_mn, nloc_max_mn, mlocW, mloc_pinv, nloc_pinv);
       
        if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "Desc Init starts %d\n", mloc);

        /*
         * Allocatation for QDWHPartial
         */
        sizeS = 0, k = 0;
        descinit_( descA, &m, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
        descinit_( descB, &ldw, &min_mn, &nb, &nb, &i0, &i0, &ictxt, &mlocW, &info );
        descinit_( descU, &m, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
        //descinit_( descU, &m, &min_mn, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );
        descinit_( descVT, &n, &min_mn, &nb, &nb, &i0, &i0, &ictxt, &mloc_n, &info );
        descinit_( descpinv, &n, &m, &nb, &nb, &i0, &i0, &ictxt, &mloc_pinv, &info );

        if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "Desc Init ends %d\n", mloc);

        A     = (double *)malloc(mloc*nloc*sizeof(double));
        B     = (double *)malloc(mlocW*nloc_min_mn*sizeof(double));
        //U     = (double *)malloc(mloc*nloc_min_mn*sizeof(double)) ;
        U     = (double *)malloc(mloc*nloc*sizeof(double));
        VT    = (double *)malloc(mloc_n*nloc_min_mn*sizeof(double));
        pinv  = (double *)malloc(mloc_pinv*nloc_pinv*sizeof(double));
        S     = (double *)malloc(min_mn*sizeof(double));

        /*
         * Allocatation to check the results
        float *A, *Acpy, *test_pinv1, *test_pinv2, *pinv;
        magma_smalloc_pinned( &A, m*n );
        magma_smalloc_pinned( &Acpy, m*n );
        magma_smalloc_pinned( &test_pinv1, m*m );
        magma_smalloc_pinned( &test_pinv2, m*n );
        magma_smalloc_pinned( &pinv, n*m );

        lapackf77_slaset( "A", &m, &n, &alpha, &alpha, A, &m );
        lapackf77_slaset( "A", &m, &n, &alpha, &alpha, Acpy, &m );
        */
        Acpy    = (double *)malloc(mloc*nloc*sizeof(double)) ;
        descinit_( descAcpy, &m, &n, &nb, &nb, &i0, &i0, &ictxt, &mloc, &info );

        /*
         * Allocatation for DLATMS
         */
        D     = (double *)malloc(min_mn*sizeof(double)) ;
        W     = (double *)malloc(min_mn*sizeof(double)) ;
        char   *dist = "N"; /* NORMAL( 0, 1 )  ( 'N' for normal ) */
        int    iseed[4] = {1, 0, 0, 1};
        char   *sym = "P"; /* The generated matrix is symmetric, with
                             eigenvalues (= singular values) specified by D, COND,
                             MODE, and DMAX; they will not be negative.
                             "N" not supported. */
        //int    mode = 4; /* sets D(i)=1 - (i-1)/(N-1)*(1 - 1/COND) */
        //double cond = 1.0/eps;
        double dmax = 1.0;
        int    kl   = n;
        int    ku   = n;
        char   *pack = "N"; /* no packing */
        int    order = n;
        int    info;

        pdlasizesep_( descA,
                      &iprepad, &ipostpad, &sizemqrleft, &sizemqrright, &sizeqrf,
                      &lwork,
                      &sizeqtq, &sizechk, &sizesyevx, &isizesyevx, &sizesubtst,
                      &isizesubtst, &sizetst, &isizetst );
        if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "Setting lwork done\n");
        Wloc1 = (double *)calloc(max(m,lwork),sizeof(double)) ;

        /*
         * For matrix similar to matlab
         * Generate random synthetic matrices using SLATMS
         * Condition number = sigma_max / Sigma_min (if the mode = 0)
         */
        mode = 0; 
        //double cond = 1.0/eps*1.e0;
        
        if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "Setting singular values \n");
        if ( myrank_mpi == 0 ) printf("\n min_mn %d \n", min_mn);
        for( i = 0; i < min_mn; i++ ){
           W[i] =  pow(0.5, (double)i/(double)min_mn*100.); // geometrically increasing
        }
        double cond = W[0] / W[min_mn-1];
        if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "Setting singular values done\n");

        pdlatms_(&m, &n, dist,
                 iseed, sym, W, &mode, &cond, &dmax,
                 &m, &n, pack,
                 A, &i1, &i1, descA, &n,
                 Wloc1, &lwork, &info);
        if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "MatGen done\n");
        if ( info != 0 ) {
            fprintf(stderr, "An error occured during matrix generation: %d\n", info );
            return EXIT_FAILURE;
        }

        /*
         * Save copy of the matrix
         */
        pdlacpy_( "A", &m, &n,
                   A, &i1, &i1, descA,
                   Acpy, &i1, &i1, descAcpy );


        {
        /*
         * Find SVD using SGESVD  
        double *Wsvd, *Usvd, *Vsvd;
        int descUsvd[9], descVsvd[9];
        int lworksvd = -1;
        Wsvd = (double *)malloc(1*sizeof(double));
        pdgesvd_( "N", "N", &m, &n, A, &i1, &i1, descA,
                  W,
                  Usvd, &i1, &i1, descUsvd,
                  Vsvd, &i1, &i1, descVsvd,
                  Wsvd, &lworksvd, &my_info_facto );
        lworksvd = Wsvd[0];
        Wsvd = (double *)malloc(lworksvd*sizeof(double));

        pdgesvd_( "N", "N", &m, &n, A, &i1, &i1, descA,
                  W,
                  Usvd, &i1, &i1, descUsvd,
                  Vsvd, &i1, &i1, descVsvd,
                  Wsvd, &lworksvd, &my_info_facto );


        free(Wsvd);
         */
        }
     
        double normA  = pdlange_ ( "f", &m, &n, Acpy, &i1, &i1, descAcpy, Wloc1);
        // loop over iteration
        for ( iter = 0; iter < niter; iter++ ) {
              if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "\nScaLAPACK dgesvd starts...\n");

                 my_elapsed_qwsvd = 0.0, elapsed_qwsvd = 0.0;
                 my_elapsed_pdgesvd = 0.0, elapsed_pdgesvd = 0.0;
                 /*
                  * Save copy of A in Acpy  
                  */
                 pdlacpy_( "A", &m, &n,
                           Acpy,    &i1, &i1, descAcpy,
                           A, &i1, &i1, descA );


                 /************************************************************
                  * Call QDWHPartial  
                  ************************************************************/
                 {
                 double s   = 1.e-4; //Threshold for the singular values
                 double tol = 1.e-2; //Tolerance (governs accuracy) 
                 int it = 0;
                 int fact = 1;
                 int psinv = 0;
                 flops = 0.0;

                 if ( myrank_mpi == 0 ) printf("\n Start QDWHpartial ... \n");
                 my_elapsed_qwsvd   = 0.0;
                 my_elapsed_qwsvd   =- MPI_Wtime();


                   QDWHpartial ( m, n, // Size of matrix
                                 fact, psinv,
                                 s, // Threshold
                                 tol, // Tolerance (governs accuracy) 
                                 A,  i1, i1, descA, // Matrix
                                 S,          // Sigular values, size min(m,n)
                                 U,  i1, i1, descU, // Left singular vectors, size mx(10%n)
                                 VT, i1, i1, descVT, // Right singular vectors, size nxn, d_VT = d_VT 
                                 B,  i1, i1, descB,// Needed for the QR fact in QDWH, it is of size NxN, because the matrix will reduced 
                                 &sizeS, // Num-computed-SV
                                 &k, // Num-wanted-SV
                                 &it,
                                 &flops); 



                 my_elapsed_qwsvd   += MPI_Wtime();
                 MPI_Allreduce( &my_elapsed_qwsvd, &elapsed_qwsvd, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                 sumtime_qwsvd += elapsed_qwsvd;
                 if ( elapsed_qwsvd >= max_time_qwsvd ) { max_time_qwsvd = elapsed_qwsvd;}
                 if ( elapsed_qwsvd <= min_time_qwsvd ) { min_time_qwsvd = elapsed_qwsvd;}
                 if ( myrank_mpi == 0 ) printf("\n Done QDWHpartial ... \n");

                 ktmp = k; 
                 sizeStmp = sizeS;
                 k = min(ktmp,sizeStmp);  
                 sizeS = k;

                 /*
                  * Check the results
                  */
                 if ( check )
                 {
                    if ( myrank_mpi == 0 ) {
                    fprintf(stderr, "\n QDWHpartial \n");
                    fprintf(stderr, "/////////////////////////////////////////////////////////////////////////\n");
                    }

                    for (i = 0; i < k; i++){
                        D[i] = W[i] - S[i];
                    }
                    alpha = 1.0; beta = -1.0;

                    int ione = 1;
                    double acc_sv  = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'f', k, ione, D, ione, Wloc1);
                    double norm_sv = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'f', k, ione, W, ione, Wloc1);

                    if ( myrank_mpi == 0 ) {
                    fprintf(stderr, "/////////////////////////////////////////////////////////////////////////\n");
                    }


                    double *ResR = (double *)malloc(k*sizeof(double));
                    double *ResL = (double *)malloc(k*sizeof(double));
                    double max_resR_err, max_resL_err;

                    //double *VT, *U;
                    //VT = (double *)malloc(n*sizeS*sizeof(double)) ;
                    //U  = (double *)malloc(m*sizeS*sizeof(double)) ;


                    /* checking orthogonality U */
                    alpha = 0.0; beta = 1.0;
                    pdlaset_( "G", &k, &k, 
                              &alpha, &beta, 
                              A, &i1, &i1, descA);
                    alpha = 1.0; beta = -1.0;
                    pdgemm_( "T", "N", 
                             &k, &k, &m,
                             &alpha, U, &i1, &i1, descU,
                                     U, &i1, &i1, descU,
                             &beta,  A, &i1, &i1, descA);
                    double orth_U  = pdlange_ ( "f", &k, &k, A, &i1, &i1, descA, Wloc1);

                    /* checking orthogonality VT */
                    alpha = 0.0; beta = 1.0;
                    pdlaset_( "G", &k, &k, 
                              &alpha, &beta, 
                              A, &i1, &i1, descA);
                    alpha = 1.0; beta = -1.0;
                    pdgemm_( "T", "N", 
                             &k, &k, &n,
                             &alpha, VT, &i1, &i1, descVT,
                                     VT, &i1, &i1, descVT,
                             &beta,  A,  &i1, &i1, descA);
                    double orth_VT = pdlange_ ( "f", &k, &k, A, &i1, &i1, descA, Wloc1);

                    /* 
                     * checking Berr = A-USVT 
                     * res(ii) = norm(   A*V0(:,ii)-U0(:,ii)*S0(ii,ii)         );
                     * resL(ii) = norm(U0(:,ii)'*A-V0(:,ii)'*S0(ii,ii));    
                     */

                    /* Right Residual */
                    alpha = 1.0; beta = 0.0;
                    pdgemv_ ("N", &m, &n, &alpha, Acpy, &i1, &i1, descAcpy, VT, &i1, &i1, descVT, &i1, &beta, A, &i1, &i1, descA, &i1);
                    alpha = 0.0; beta = 1.0;
                    pdlaset_( "G", &m, &n, 
                              &alpha, &beta, 
                              pinv, &i1, &i1, descpinv);
                    alpha = -S[0]; beta = 1.0;
                    pdgemv_ ("N", &m, &n, &alpha, pinv, &i1, &i1, descpinv, U, &i1, &i1, descU, &i1, &beta, A, &i1, &i1, descA, &i1);
                    ResR[0] = pdlange_ ( "f", &m, &ione, A, &i1, &i1, descA, Wloc1);

                    int i1_i;
                    max_resR_err = ResR[0];
                    for (i = 1; i < k; i++){
                        i1_i = i1 + i;
                        alpha = 1.0; beta = 0.0;
                        pdgemv_ ("N", &m, &n, &alpha, Acpy, &i1, &i1, descAcpy, VT, &i1, &i1_i, descVT, &i1, &beta, A, &i1, &i1, descA, &i1);
                        alpha = 0.0; beta = 1.0;
                        pdlaset_( "G", &m, &n, 
                                  &alpha, &beta, 
                                  pinv, &i1, &i1, descpinv);
                        alpha = -S[i]; beta = 1.0;
                        pdgemv_ ("N", &m, &n, &alpha, pinv, &i1, &i1, descpinv, U, &i1, &i1_i, descU, &i1, &beta, A, &i1, &i1, descA, &i1);
                        ResR[i] = pdlange_ ( "f", &m, &ione, A, &i1, &i1, descA, Wloc1);
                        if ( ResR[i] > max_resR_err ) { max_resR_err = ResR[i];}
                    }


                    /* Left Residual */
                    alpha = 1.0; beta = 0.0;
                    pdgemv_ ("N", &m, &n, &alpha, Acpy, &i1, &i1, descAcpy, U, &i1, &i1, descU, &i1, &beta, A, &i1, &i1, descA, &i1);
                    alpha = 0.0; beta = 1.0;
                    pdlaset_( "G", &m, &n, 
                              &alpha, &beta, 
                              pinv, &i1, &i1, descpinv);
                    alpha = -S[0]; beta = 1.0;
                    pdgemv_ ("N", &m, &n, &alpha, pinv, &i1, &i1, descpinv, VT, &i1, &i1, descVT, &i1, &beta, A, &i1, &i1, descA, &i1);
                    ResL[0] = pdlange_ ( "f", &m, &ione, A, &i1, &i1, descA, Wloc1);

                    max_resL_err = ResL[0];
                    for (i = 1; i < k; i++){
                        i1_i = i1 + i;
                        alpha = 1.0; beta = 0.0;
                        pdgemv_ ("N", &m, &n, &alpha, Acpy, &i1, &i1, descAcpy, U, &i1, &i1_i, descU, &i1, &beta, A, &i1, &i1, descA, &i1);
                        alpha = 0.0; beta = 1.0;
                        pdlaset_( "G", &m, &n, 
                                  &alpha, &beta, 
                                  pinv, &i1, &i1, descpinv);
                        alpha = -S[i]; beta = 1.0;
                        pdgemv_ ("N", &m, &n, &alpha, pinv, &i1, &i1, descpinv, VT, &i1, &i1_i, descVT, &i1, &beta, A, &i1, &i1, descA, &i1);
                        ResL[i] = pdlange_ ( "f", &n, &ione, A, &i1, &i1, descA, Wloc1);
                        if ( ResL[i] > max_resL_err ) { max_resL_err = ResL[i];}
                    }


                    if ( myrank_mpi == 0 ){
                         fprintf(stderr, "  M  \tN  \tNB  \tNP  \tP  \tQ  \t#It \tCond   \tTime \t\tGFlops/s \tThreshold \tNum-wanted-SV \tNum-computed-SV Acc-SV \tOrth-U \tOrth-VT  ResR \tResL\n"
                         "  %d  %d  \t%4d \t%4d \t%3d \t%3d \t%d %2.4e \t%7.2f  \t%7.2f  \t%2.4e  \t%d    \t\t%d   \t\t%2.1e  %2.1e %2.1e  %2.1e %2.1e\n",
                         //m, n, it, cond, timer, flops/timer/1.e9, s, ktmp, sizeStmp, acc_sv/norm_sv, orth_U/m, orth_VT/n, max_resR_err/(normA*max(m,n)), max_resL_err/(normA*max(m,n)));
                         m, n, nb, nprocs_mpi, nprow, npcol, it, cond, elapsed_qwsvd, flops/elapsed_qwsvd/1.e9, s, ktmp, sizeStmp, acc_sv/norm_sv, orth_U/k, orth_VT/k, max_resR_err/(normA*k), max_resL_err/(normA*k));
                    }




                 }//End of checking the results
                 else
                 {
                    if ( myrank_mpi == 0 ){
                         fprintf(stderr, "  M  \tN  \tNB  \tNP  \tP  \tQ  \t#It \tCond   \tTime \t\tGFlops/s \tThreshold \tNum-wanted-SV \tNum-computed-SV Acc-SV \tOrth-U \tOrth-VT  ResR \tResL\n"
                         "  %d  %d  \t%4d \t%4d \t%3d \t%3d \t%d %2.4e \t%7.2f  \t%7.2f  \t%2.4e  \t%d    \t\t%d   \t\t%2.1e  %2.1e %2.1e  %2.1e %2.1e\n",
                         //m, n, it, cond, timer, flops/timer/1.e9, s, ktmp, sizeStmp, acc_sv/norm_sv, orth_U/m, orth_VT/n, max_resR_err/(normA*max(m,n)), max_resL_err/(normA*max(m,n)));
                         m, n, nb, nprocs_mpi, nprow, npcol, it, cond, elapsed_qwsvd, flops/elapsed_qwsvd/1.e9, s, ktmp, sizeStmp, 0.0, 0.0, 0.0, 0.0, 0.0);
                    }
                 }

                 }//End of QDWHpartial call
                 /************************************************************
                  *  End QDWHpartial 
                  ************************************************************/





                 /************************************************************
                  * Call PDGESVD 
                  ************************************************************/
                 if( 0 )
                 {
                 lwork_pdgesvd = -1;
                 Work_pdgesvd  = (double *)calloc(1,sizeof(double));

                 pdgesvd_( "V", "V", &m, &n, 
                           A, &i1, &i1, descA,
                           S,
                           U, &i1, &i1, descU,
                           VT, &i1, &i1, descVT,
                           Work_pdgesvd, &lwork_pdgesvd, &info);

                 lwork_pdgesvd = (int)Work_pdgesvd[0];
                 Work_pdgesvd  = (double *)calloc(lwork_pdgesvd,sizeof(double));

                 pdlacpy_( "A", &m, &n,
                            Acpy, &i1, &i1, descAcpy,
                            A,    &i1, &i1, descA );

                 if ( myrank_mpi == 0 ) printf("\n Start PDGESVD ... \n");
                 my_elapsed_pdgesvd   = 0.0;
                 my_elapsed_pdgesvd   =- MPI_Wtime();

                 pdgesvd_( "V", "V", &m, &n, 
                           A, &i1, &i1, descA,
                           S,
                           U, &i1, &i1, descU,
                           VT, &i1, &i1, descVT,
                           Work_pdgesvd, &lwork_pdgesvd, &info );

                 my_elapsed_pdgesvd   += MPI_Wtime();
                 MPI_Allreduce( &my_elapsed_pdgesvd, &elapsed_pdgesvd, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                 sumtime_pdgesvd += elapsed_pdgesvd;
                 if ( elapsed_pdgesvd >= max_time_pdgesvd ) { max_time_pdgesvd = elapsed_pdgesvd;}
                 if ( elapsed_pdgesvd <= min_time_pdgesvd ) { min_time_pdgesvd = elapsed_pdgesvd;}
                 if ( myrank_mpi == 0 ) printf("\n Done PDGESVD ... \n");
                 long int min_Ns = min_mn * min_mn * min_mn;
                 long int max_Ns = max_mn * min_mn * min_mn;
                 if ( m!=n ){
                     // = (2*max_mn*min_mn^2      - 2/3*min_mn^3)         + (22*min_mn^3)          + 2*min_mn^3
                     flops_pdgesvd += 2.*(double)max_Ns - 
                                      2./3.*(double)min_Ns + 
                                      22.*(double)min_Ns + 
                                      2.*(double)min_Ns;
                 }
                 else{
                     flops_pdgesvd += 22.*(double)min_Ns;
                 }


                 /*
                  * Check the results
                  */
                 if ( check )
                 {
                    if ( myrank_mpi == 0 ) {
                    fprintf(stderr, "\n PDGESVD \n");
                    fprintf(stderr, "/////////////////////////////////////////////////////////////////////////\n");
                    }

                    dlasrt_( "D", &n, W, &info );
                    dlasrt_( "D", &n, S, &info );

                    for (i = 0; i < k; i++){
                        D[i] = W[i] - S[i];
                    }

                    alpha = 1.0; beta = -1.0;

                    int ione = 1;
                    double acc_sv  = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'f', k, ione, D, ione, Wloc1);
                    double norm_sv = LAPACKE_dlange_work( LAPACK_COL_MAJOR, 'f', k, ione, W, ione, Wloc1);

                    if ( myrank_mpi == 0 ) {
                    fprintf(stderr, "/////////////////////////////////////////////////////////////////////////\n");
                    }


                    double *ResR = (double *)malloc(k*sizeof(double));
                    double *ResL = (double *)malloc(k*sizeof(double));
                    double max_resR_err, max_resL_err;

                    //double *VT, *U;
                    //VT = (double *)malloc(n*sizeS*sizeof(double)) ;
                    //U  = (double *)malloc(m*sizeS*sizeof(double)) ;


                    /* checking orthogonality U */
                    alpha = 0.0; beta = 1.0;
                    pdlaset_( "G", &k, &k, 
                              &alpha, &beta, 
                              A, &i1, &i1, descA);
                    alpha = 1.0; beta = -1.0;
                    pdgemm_( "T", "N", 
                             &k, &k, &m,
                             &alpha, U, &i1, &i1, descU,
                                     U, &i1, &i1, descU,
                             &beta,  A, &i1, &i1, descA);
                    double orth_U  = pdlange_ ( "f", &k, &k, A, &i1, &i1, descA, Wloc1);

                    /* checking orthogonality VT */
                    alpha = 0.0; beta = 1.0;
                    pdlaset_( "G", &k, &k, 
                              &alpha, &beta, 
                              A, &i1, &i1, descA);
                    alpha = 1.0; beta = -1.0;
                    pdgemm_( "T", "N", 
                             &k, &k, &n,
                             &alpha, VT, &i1, &i1, descVT,
                                     VT, &i1, &i1, descVT,
                             &beta,  A,  &i1, &i1, descA);
                    double orth_VT = pdlange_ ( "f", &k, &k, A, &i1, &i1, descA, Wloc1);

                    /* 
                     * checking Berr = A-USVT 
                     * res(ii) = norm(   A*V0(:,ii)-U0(:,ii)*S0(ii,ii)         );
                     * resL(ii) = norm(U0(:,ii)'*A-V0(:,ii)'*S0(ii,ii));    
                     */

                    /* Right Residual */
                    alpha = 1.0; beta = 0.0;
                    pdgemv_ ("N", &m, &n, &alpha, Acpy, &i1, &i1, descAcpy, VT, &i1, &i1, descVT, &n, &beta, A, &i1, &i1, descA, &i1);
                    alpha = 0.0; beta = 1.0;
                    pdlaset_( "G", &m, &n, 
                              &alpha, &beta, 
                              pinv, &i1, &i1, descpinv);
                    alpha = -S[0]; beta = 1.0;
                    pdgemv_ ("N", &m, &n, &alpha, pinv, &i1, &i1, descpinv, U, &i1, &i1, descU, &i1, &beta, A, &i1, &i1, descA, &i1);
                    ResR[0] = pdlange_ ( "f", &m, &ione, A, &i1, &i1, descA, Wloc1);

                    int i1_i;
                    max_resR_err = ResR[0];
                    for (i = 1; i < k; i++){
                        i1_i = i1 + i;
                        alpha = 1.0; beta = 0.0;
                        pdgemv_ ("N", &m, &n, &alpha, Acpy, &i1, &i1, descAcpy, VT, &i1_i, &i1, descVT, &n, &beta, A, &i1, &i1, descA, &i1);
                        alpha = 0.0; beta = 1.0;
                        pdlaset_( "G", &m, &n, 
                                  &alpha, &beta, 
                                  pinv, &i1, &i1, descpinv);
                        alpha = -S[i]; beta = 1.0;
                        pdgemv_ ("N", &m, &n, &alpha, pinv, &i1, &i1, descpinv, U, &i1, &i1_i, descU, &i1, &beta, A, &i1, &i1, descA, &i1);
                        ResR[i] = pdlange_ ( "f", &m, &ione, A, &i1, &i1, descA, Wloc1);
                        if ( ResR[i] > max_resR_err ) { max_resR_err = ResR[i];}
                    }


                    /* Left Residual */
                    alpha = 1.0; beta = 0.0;
                    pdgemv_ ("N", &m, &n, &alpha, Acpy, &i1, &i1, descAcpy, U, &i1, &i1, descU, &i1, &beta, A, &i1, &i1, descA, &i1);
                    alpha = 0.0; beta = 1.0;
                    pdlaset_( "G", &m, &n, 
                              &alpha, &beta, 
                              pinv, &i1, &i1, descpinv);
                    alpha = -S[0]; beta = 1.0;
                    pdgemv_ ("N", &m, &n, &alpha, pinv, &i1, &i1, descpinv, VT, &i1, &i1, descVT, &n, &beta, A, &i1, &i1, descA, &i1);
                    ResL[0] = pdlange_ ( "f", &m, &ione, A, &i1, &i1, descA, Wloc1);

                    max_resL_err = ResL[0];
                    for (i = 1; i < k; i++){
                        i1_i = i1 + i;
                        alpha = 1.0; beta = 0.0;
                        pdgemv_ ("N", &m, &n, &alpha, Acpy, &i1, &i1, descAcpy, U, &i1, &i1_i, descU, &i1, &beta, A, &i1, &i1, descA, &i1);
                        alpha = 0.0; beta = 1.0;
                        pdlaset_( "G", &m, &n, 
                                  &alpha, &beta, 
                                  pinv, &i1, &i1, descpinv);
                        alpha = -S[i]; beta = 1.0;
                        pdgemv_ ("N", &m, &n, &alpha, pinv, &i1, &i1, descpinv, VT, &i1_i, &i1, descVT, &n, &beta, A, &i1, &i1, descA, &i1);
                        ResL[i] = pdlange_ ( "f", &n, &ione, A, &i1, &i1, descA, Wloc1);
                        if ( ResL[i] > max_resL_err ) { max_resL_err = ResL[i];}
                    }


                    if ( myrank_mpi == 0 ){
                         fprintf(stderr, "  M  \tN  \tNB  \tNP  \tP  \tQ  \t#It \tCond   \tTime \t\tGFlops/s \tThreshold \tNum-wanted-SV \tNum-computed-SV Acc-SV \tOrth-U \tOrth-VT  ResR \tResL\n"
                         "  %d  %d  \t%4d \t%4d \t%3d \t%3d \t%d %2.4e \t%7.2f  \t%7.2f  \t%2.4e  \t%d    \t\t%d   \t\t%2.1e  %2.1e %2.1e  %2.1e %2.1e\n",
                         m, n, nb, nprocs_mpi, nprow, npcol, 0, cond, elapsed_pdgesvd, flops_pdgesvd/elapsed_pdgesvd/1.e9, 0.0, ktmp, sizeStmp, acc_sv/norm_sv, orth_U/k, orth_VT/k, max_resR_err/(normA*k), max_resL_err/(normA*k));
                    }


                 }//End of checking the results
                 else
                 {
                    if ( myrank_mpi == 0 ){
                         fprintf(stderr, "  M  \tN  \tNB  \tNP  \tP  \tQ  \t#It \tCond   \tTime \t\tGFlops/s \tThreshold \tNum-wanted-SV \tNum-computed-SV Acc-SV \tOrth-U \tOrth-VT  ResR \tResL\n"
                         "  %d  %d  \t%4d \t%4d \t%3d \t%3d \t%d %2.4e \t%7.2f  \t%7.2f  \t%2.4e  \t%d    \t\t%d   \t\t%2.1e  %2.1e %2.1e  %2.1e %2.1e\n",
                         m, n, nb, nprocs_mpi, nprow, npcol, 0, cond, elapsed_pdgesvd, flops_pdgesvd/elapsed_pdgesvd/1.e9, 0.0, ktmp, sizeStmp, 0.0, 0.0, 0.0, 0.0, 0.0);
                    }
                 }

                 }//End of PDGESVD call
                 /************************************************************
                  *  End PDGESVD 
                  ************************************************************/


        } // loop over iteration
        free(A); free(B); free(U); free(VT); free(pinv); free(S); free(Acpy); free(D); free(W);
    } // End loop over range
    if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "Range loop ends\n");

    blacs_gridexit_( &i0 );
    MPI_Finalize();
    if ( verbose & myrank_mpi == 0 ) fprintf(stderr, "Program ends...\n");
    return 0;

}

int
get_range(char *range, int *start_p, int *stop_p, int *step_p) {
    char *s, *s1, buf[21];
    int colon_count, copy_len, nbuf=20, n;
    int start=1000, stop=10000, step=1000;

    colon_count = 0;
    for (s = strchr( range, ':'); s; s = strchr( s+1, ':'))
        colon_count++;

    if (colon_count == 0) { /* No colon in range. */
        if (sscanf( range, "%d", &start ) < 1 || start < 1)
            return -1;
        step = start / 10;
        if (step < 1) step = 1;
        stop = start + 10 * step;

    } else if (colon_count == 1) { /* One colon in range.*/
        /* First, get the second number (after colon): the stop value. */
        s = strchr( range, ':' );
        if (sscanf( s+1, "%d", &stop ) < 1 || stop < 1)
            return -1;

        /* Next, get the first number (before colon): the start value. */
        n = s - range;
        copy_len = n > nbuf ? nbuf : n;
        strncpy( buf, range, copy_len );
        buf[copy_len] = 0;
        if (sscanf( buf, "%d", &start ) < 1 || start > stop || start < 1)
            return -1;

        /* Let's have 10 steps or less. */
        step = (stop - start) / 10;
        if (step < 1)
            step = 1;
    } else if (colon_count == 2) { /* Two colons in range. */
        /* First, get the first number (before the first colon): the start value. */
        s = strchr( range, ':' );
        n = s - range;
        copy_len = n > nbuf ? nbuf : n;
        strncpy( buf, range, copy_len );
        buf[copy_len] = 0;
        if (sscanf( buf, "%d", &start ) < 1 || start < 1)
            return -1;

        /* Next, get the second number (after the first colon): the stop value. */
        s1 = strchr( s+1, ':' );
        n = s1 - (s + 1);
        copy_len = n > nbuf ? nbuf : n;
        strncpy( buf, s+1, copy_len );
        buf[copy_len] = 0;
        if (sscanf( buf, "%d", &stop ) < 1 || stop < start)
            return -1;

        /* Finally, get the third number (after the second colon): the step value. */
        if (sscanf( s1+1, "%d", &step ) < 1 || step < 1)
            return -1;
    } else

        return -1;

    *start_p = start;
    *stop_p = stop;
    *step_p = step;

    return 0;
}

