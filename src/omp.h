#ifndef omp_h
#define omp_h

#if defined(_OPENMP)
#include <omp.h>
#else
#pragma message("Warning: OpenMP is not available")
inline void omp_set_dynamic(int flag) {}
inline int omp_get_max_threads() { return 1;}
inline int omp_get_num_procs() { return 1; }
inline void omp_set_num_threads(int nthread){}
inline int omp_get_thread_num() {return 0; };
#endif
#endif /* omp_h */
