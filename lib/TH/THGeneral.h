#ifndef TH_GENERAL_INC
#define TH_GENERAL_INC

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <time.h>
#include <string.h>
#include <stddef.h>

#ifdef __cplusplus
# define TH_EXTERNC extern "C"
#else
# define TH_EXTERNC extern
#endif

#ifdef _WIN32
# ifdef TH_EXPORTS
#  define TH_API TH_EXTERNC __declspec(dllexport)
# else
#  define TH_API TH_EXTERNC __declspec(dllimport)
# endif
#else
# define TH_API TH_EXTERNC
#endif

TH_API void _THError(const char *file, const int line, const char *fmt, ...);
TH_API void _THAssertionFailed(const char *file, const int line, const char *exp, const char *fmt, ...);
TH_API void _THArgCheck(const char *file, int line, int condition, int argNumber, const char *fmt, ...);
TH_API void* THRealloc(void *ptr, ptrdiff_t size);
TH_API void* THAlloc(ptrdiff_t size);
TH_API void THFree(void *ptr);

#define THError(...) _THError(__FILE__, __LINE__, __VA_ARGS__)

#define THCleanup(...) __VA_ARGS__

#define THArgCheck(...)                                               \
do {                                                                  \
  _THArgCheck(__FILE__, __LINE__, __VA_ARGS__);                       \
} while(0)

#define THArgCheckWithCleanup(condition, cleanup, ...)                \
do if (!(condition)) {                                                \
  cleanup                                                             \
  _THArgCheck(__FILE__, __LINE__, 0, __VA_ARGS__);                    \
} while(0)

#define THAssert(exp)                                                 \
do {                                                                  \
  if (!(exp)) {                                                       \
    _THAssertionFailed(__FILE__, __LINE__, #exp, "");                 \
  }                                                                   \
} while(0)

#define THAssertMsg(exp, ...)                                         \
do {                                                                  \
  if (!(exp)) {                                                       \
    _THAssertionFailed(__FILE__, __LINE__, #exp, __VA_ARGS__);        \
  }                                                                   \
} while(0)

#endif
