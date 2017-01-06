#ifndef TH_FILE_INC
#define TH_FILE_INC

#include "THGeneral.h"

typedef struct THFile__ THFile;

TH_API int THFile_isOpened(THFile *self);
TH_API int THFile_isQuiet(THFile *self);
TH_API int THFile_isReadable(THFile *self);
TH_API int THFile_isWritable(THFile *self);
TH_API int THFile_isBinary(THFile *self);
TH_API int THFile_isAutoSpacing(THFile *self);
TH_API int THFile_hasError(THFile *self);

TH_API void THFile_binary(THFile *self);
TH_API void THFile_ascii(THFile *self);
TH_API void THFile_autoSpacing(THFile *self);
TH_API void THFile_noAutoSpacing(THFile *self);
TH_API void THFile_quiet(THFile *self);
TH_API void THFile_pedantic(THFile *self);
TH_API void THFile_clearError(THFile *self);

/* scalar */
TH_API unsigned char THFile_readByteScalar(THFile *self);
TH_API char THFile_readCharScalar(THFile *self);
TH_API short THFile_readShortScalar(THFile *self);
TH_API int THFile_readIntScalar(THFile *self);
TH_API long THFile_readLongScalar(THFile *self);
TH_API float THFile_readFloatScalar(THFile *self);
TH_API double THFile_readDoubleScalar(THFile *self);

/* raw */
TH_API size_t THFile_readByteRaw(THFile *self, unsigned char *data, size_t n);
TH_API size_t THFile_readCharRaw(THFile *self, char *data, size_t n);
TH_API size_t THFile_readShortRaw(THFile *self, short *data, size_t n);
TH_API size_t THFile_readIntRaw(THFile *self, int *data, size_t n);
TH_API size_t THFile_readLongRaw(THFile *self, long *data, size_t n);
TH_API size_t THFile_readFloatRaw(THFile *self, float *data, size_t n);
TH_API size_t THFile_readDoubleRaw(THFile *self, double *data, size_t n);
TH_API size_t THFile_readStringRaw(THFile *self, const char *format, char **str_); /* you must deallocate str_ */

TH_API void THFile_synchronize(THFile *self);
TH_API void THFile_seek(THFile *self, size_t position);
TH_API void THFile_seekEnd(THFile *self);
TH_API size_t THFile_position(THFile *self);
TH_API void THFile_close(THFile *self);
TH_API void THFile_free(THFile *self);

#endif
