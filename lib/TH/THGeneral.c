#include "THGeneral.h"

#if (defined(__unix) || defined(_WIN32))
  #if defined(__FreeBSD__)
    #include <malloc_np.h>
  #else
    #include <malloc.h>
  #endif
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#endif

/* Torch Error Handling */
static void defaultErrorHandlerFunction(const char *msg)
{
  printf("$ Error: %s\n", msg);
  exit(-1);
}

void _THError(const char *file, const int line, const char *fmt, ...)
{
  char msg[2048];
  va_list args;

  /* vasprintf not standard */
  /* vsnprintf: how to handle if does not exists? */
  va_start(args, fmt);
  int n = vsnprintf(msg, 2048, fmt, args);
  va_end(args);

  if(n < 2048) {
    snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
  }

  defaultErrorHandlerFunction(msg);
}

void _THAssertionFailed(const char *file, const int line, const char *exp, const char *fmt, ...) {
  char msg[1024];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, 1024, fmt, args);
  va_end(args);
  _THError(file, line, "Assertion `%s' failed. %s", exp, msg);
}

void _THArgCheck(const char *file, int line, int condition, int argNumber, const char *fmt, ...)
{
  if(!condition) {
    char msg[2048];
    va_list args;

    /* vasprintf not standard */
    /* vsnprintf: how to handle if does not exists? */
    va_start(args, fmt);
    int n = vsnprintf(msg, 2048, fmt, args);
    va_end(args);

    if(n < 2048) {
      snprintf(msg + n, 2048 - n, " at %s:%d", file, line);
    }

    defaultErrorHandlerFunction(msg);
  }
}

void* THRealloc(void *ptr, ptrdiff_t size)
{
  return realloc(ptr, size);
}

void* THAlloc(ptrdiff_t size)
{
  return malloc(size);
}

void THFree(void *ptr)
{
  free(ptr);
}
