struct THFile__
{
    struct THFileVTable *vtable;

    int isQuiet;
    int isReadable;
    int isWritable;
    int isBinary;
    int isAutoSpacing;
    int hasError;
};

/* virtual table definition */

struct THFileVTable
{
    int (*isOpened)(THFile *self);

    size_t (*readByte)(THFile *self, unsigned char *data, size_t n);
    size_t (*readChar)(THFile *self, char *data, size_t n);
    size_t (*readShort)(THFile *self, short *data, size_t n);
    size_t (*readInt)(THFile *self, int *data, size_t n);
    size_t (*readLong)(THFile *self, long *data, size_t n);
    size_t (*readFloat)(THFile *self, float *data, size_t n);
    size_t (*readDouble)(THFile *self, double *data, size_t n);
    size_t (*readString)(THFile *self, const char *format, char **str_);

    void (*synchronize)(THFile *self);
    void (*seek)(THFile *self, size_t position);
    void (*seekEnd)(THFile *self);
    size_t (*position)(THFile *self);
    void (*close)(THFile *self);
    void (*free)(THFile *self);
};
