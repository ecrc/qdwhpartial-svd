#include <unistd.h>
#include <sys/time.h>
#include <sys/resource.h>
//#include <plasma.h>

//#define QUARK_Insert_Task QUARK_Execute_Task

#define USAGE(name, args, details)                  \
  printf(" Proper Usage is : ./exe " args " with\n" \
         "  " name "\n"   \
         details);
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif


static inline double cWtime(void)
{
    struct timeval tp;
    gettimeofday( &tp, NULL );
    return tp.tv_sec + 1e-6 * tp.tv_usec;
}

