SCALAPACKDIR=/project/k1124/sukkarde/codes/scalapack_installer-intel/install
CRAYSCIDIR=/opt/cray/pe/libsci/17.12.1/INTEL/16.0/haswell

MPICC = cc
MPIF77 = ftn
CFLAGS = -Wall -Ofast -O3 -m64 -DHAVE_GETOPT_LONG

CC = $(MPICC)
LD = ftn -Ofast -O3

# Intel ScaLAPACK: error messeage invalid tags 500000, be aware!
#LIBS = -m64 -nofor_main -mkl=parallel -L$(MKLROOT)/lib/intel64 -lmkl_scalapack_lp64 -lmkl_blacs_intelmpi_lp64 -L$(SCALAPACKDIR)/lib -lsltmg -lmkl_blacs_intelmpi_lp64 -lpthread -lm -lstdc++ -Wl,-ydgemm -Wl,-ypdpotrf -Wl,-ypdgesvd -Wl,-ypdlatms_

# Cray SCI ScaLAPACK
LIBS = -nofor_main -m64 -mkl=sequential -L$(CRAYSCIDIR)/lib -lsci_intel_mpi -lsci_intel -L$(SCALAPACKDIR)/lib -lsltmg -lpthread -lm -lstdc++ -Wl,-ydgemm_ -Wl,-ypdpotrf_ -Wl,-ypdgesvd_ -Wl,-ypdlatms_

#LIBS = -nofor_main -m64 -L$(CRAYSCIDIR)/lib -lsci_intel_mpi -mkl=sequential -L$(SCALAPACKDIR)/lib -lsltmg -lpthread -lm -lstdc++ -Wl,-ydgemm_ -Wl,-ypdpotrf_ -Wl,-ypdgesvd_ -Wl,-ypdlatms_
#LIBS = -nofor_main -m64 -L$(ELPADIR)/lib -lelpa -L$(SCALAPACKDIR)/lib -lsltmg -lpthread -lm -lstdc++ -Wl,-ydgemm_ -Wl,-ypdpotrf_ -Wl,-ypdgesvd_ -Wl,-ypdlatms_
#LIBS=-nofor_main -m64  -Wl,-ydgemm_ -Wl,-ypdpotrf_ -Wl,-ypdgesvd_ -Wl,-ypdlatms_ /project/k1124/sukkarde/codes/scalapack_installer-intel/install/lib/libsltmg.a

TARGETS = main

all: $(TARGETS)

main: main.o qdwhpartial.o pdgeqdwh.o
	$(LD) -o $@ $^ $(LIBS)

clean:
	rm -f *.o $(TARGETS)





#ftn -Ofast -O3 -o main_svd main_svd.o qdwh.o normest.o -nofor_main -m64 -L/opt/cray/libsci/16.07.1/INTEL/15.0/haswell/lib -lsci_intel_mpi -mkl=sequential -L/project/k1124/sukkarde/codes/elpa-2015.11.001-intel/install/lib -lelpa -L/project/k1124/sukkarde/codes/scalapack_installer-intel/install/lib -lsltmg -lpthread -lm -lstdc++ -Wl,-ydgemm_ -Wl,-ypdpotrf_ -Wl,-ypdgesvd_ -Wl,-ypdlatms_
#NEEDED MODULES:
#module switch PrgEnv-cray/6.0.4 PrgEnv-intel
#module load  cray-libsci/17.12.1
#module load  intel/17.0.4.196 
