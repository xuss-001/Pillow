#include "Imaging.h"

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef UINT8 pixel[4];

void static inline ImagingLineBoxBlur32(
    pixel *lineOut,
    pixel *lineIn,
    int lastx,
    int radius,
    int edgeA,
    int edgeB,
    UINT32 ww,
    UINT32 fw
) {
    int x;
    UINT32 acc[4];
    UINT32 bulk[4];

#define MOVE_ACC(acc, subtract, add)                \
    acc[0] += lineIn[add][0] - lineIn[subtract][0]; \
    acc[1] += lineIn[add][1] - lineIn[subtract][1]; \
    acc[2] += lineIn[add][2] - lineIn[subtract][2]; \
    acc[3] += lineIn[add][3] - lineIn[subtract][3];

#define ADD_FAR(bulk, acc, left, right)                                  \
    bulk[0] = (acc[0] * ww) + (lineIn[left][0] + lineIn[right][0]) * fw; \
    bulk[1] = (acc[1] * ww) + (lineIn[left][1] + lineIn[right][1]) * fw; \
    bulk[2] = (acc[2] * ww) + (lineIn[left][2] + lineIn[right][2]) * fw; \
    bulk[3] = (acc[3] * ww) + (lineIn[left][3] + lineIn[right][3]) * fw;

#define SAVE(x, bulk)                                     \
    lineOut[x][0] = (UINT8)((bulk[0] + (1 << 23)) >> 24); \
    lineOut[x][1] = (UINT8)((bulk[1] + (1 << 23)) >> 24); \
    lineOut[x][2] = (UINT8)((bulk[2] + (1 << 23)) >> 24); \
    lineOut[x][3] = (UINT8)((bulk[3] + (1 << 23)) >> 24);

    /* Compute acc for -1 pixel (outside of image):
       From "-radius-1" to "-1" get first pixel,
       then from "0" to "radius-1". */
    acc[0] = lineIn[0][0] * (radius + 1);
    acc[1] = lineIn[0][1] * (radius + 1);
    acc[2] = lineIn[0][2] * (radius + 1);
    acc[3] = lineIn[0][3] * (radius + 1);
    /* As radius can be bigger than xsize, iterate to edgeA -1. */
    for (x = 0; x < edgeA - 1; x++) {
        acc[0] += lineIn[x][0];
        acc[1] += lineIn[x][1];
        acc[2] += lineIn[x][2];
        acc[3] += lineIn[x][3];
    }
    /* Then multiply remainder to last x. */
    acc[0] += lineIn[lastx][0] * (radius - edgeA + 1);
    acc[1] += lineIn[lastx][1] * (radius - edgeA + 1);
    acc[2] += lineIn[lastx][2] * (radius - edgeA + 1);
    acc[3] += lineIn[lastx][3] * (radius - edgeA + 1);

    if (edgeA <= edgeB) {
        /* Subtract pixel from left ("0").
           Add pixels from radius. */
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        /* Subtract previous pixel from "-radius".
           Add pixels from radius. */
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc, x - radius - 1, x + radius + 1);
            SAVE(x, bulk);
        }
        /* Subtract previous pixel from "-radius".
           Add last pixel. */
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    } else {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }

#undef MOVE_ACC
#undef ADD_FAR
#undef SAVE
}

void static inline ImagingLineBoxBlur8(
    UINT8 *lineOut,
    UINT8 *lineIn,
    int lastx,
    int radius,
    int edgeA,
    int edgeB,
    UINT32 ww,
    UINT32 fw
) {
    int x;
    UINT32 acc;
    UINT32 bulk;

#define MOVE_ACC(acc, subtract, add) acc += lineIn[add] - lineIn[subtract];

#define ADD_FAR(bulk, acc, left, right) \
    bulk = (acc * ww) + (lineIn[left] + lineIn[right]) * fw;

#define SAVE(x, bulk) lineOut[x] = (UINT8)((bulk + (1 << 23)) >> 24)

    acc = lineIn[0] * (radius + 1);
    for (x = 0; x < edgeA - 1; x++) {
        acc += lineIn[x];
    }
    acc += lineIn[lastx] * (radius - edgeA + 1);

    if (edgeA <= edgeB) {
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC(acc, x - radius - 1, x + radius);
            ADD_FAR(bulk, acc, x - radius - 1, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    } else {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC(acc, 0, x + radius);
            ADD_FAR(bulk, acc, 0, x + radius + 1);
            SAVE(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC(acc, 0, lastx);
            ADD_FAR(bulk, acc, 0, lastx);
            SAVE(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC(acc, x - radius - 1, lastx);
            ADD_FAR(bulk, acc, x - radius - 1, lastx);
            SAVE(x, bulk);
        }
    }

#undef MOVE_ACC
#undef ADD_FAR
#undef SAVE
}

void static inline ImagingLineBoxBlur16(
    UINT16 *lineOut,
    const UINT16 *lineIn,
    int lastx,
    int radius,
    int edgeA,
    int edgeB,
    UINT32 ww,
    UINT32 fw
) {
    int x;
    uint64_t acc;
    uint64_t bulk;

#define MOVE_ACC16(acc, subtract, add) acc += lineIn[add] - lineIn[subtract];

#define ADD_FAR16(bulk, acc, left, right) \
    bulk = (acc * (uint64_t)ww) + (((uint64_t)lineIn[left] + lineIn[right]) * fw);

#define SAVE16(x, bulk) lineOut[x] = (UINT16)((bulk + (1 << 23)) >> 24)

    acc = (uint64_t)lineIn[0] * (radius + 1);
    for (x = 0; x < edgeA - 1; x++) {
        acc += lineIn[x];
    }
    acc += (uint64_t)lineIn[lastx] * (radius - edgeA + 1);

    if (edgeA <= edgeB) {
        for (x = 0; x < edgeA; x++) {
            MOVE_ACC16(acc, 0, x + radius);
            ADD_FAR16(bulk, acc, 0, x + radius + 1);
            SAVE16(x, bulk);
        }
        for (x = edgeA; x < edgeB; x++) {
            MOVE_ACC16(acc, x - radius - 1, x + radius);
            ADD_FAR16(bulk, acc, x - radius - 1, x + radius + 1);
            SAVE16(x, bulk);
        }
        for (x = edgeB; x <= lastx; x++) {
            MOVE_ACC16(acc, x - radius - 1, lastx);
            ADD_FAR16(bulk, acc, x - radius - 1, lastx);
            SAVE16(x, bulk);
        }
    } else {
        for (x = 0; x < edgeB; x++) {
            MOVE_ACC16(acc, 0, x + radius);
            ADD_FAR16(bulk, acc, 0, x + radius + 1);
            SAVE16(x, bulk);
        }
        for (x = edgeB; x < edgeA; x++) {
            MOVE_ACC16(acc, 0, lastx);
            ADD_FAR16(bulk, acc, 0, lastx);
            SAVE16(x, bulk);
        }
        for (x = edgeA; x <= lastx; x++) {
            MOVE_ACC16(acc, x - radius - 1, lastx);
            ADD_FAR16(bulk, acc, x - radius - 1, lastx);
            SAVE16(x, bulk);
        }
    }

#undef MOVE_ACC16
#undef ADD_FAR16
#undef SAVE16
}

Imaging
ImagingHorizontalBoxBlur(Imaging imOut, Imaging imIn, float floatRadius) {
    ImagingSectionCookie cookie;

    int y;

    int radius = (int)floatRadius;
    UINT32 ww = (UINT32)(1 << 24) / (floatRadius * 2 + 1);
    UINT32 fw = ((1 << 24) - (radius * 2 + 1) * ww) / 2;

    int edgeA = MIN(radius + 1, imIn->xsize);
    int edgeB = MAX(imIn->xsize - radius - 1, 0);

    int is16 = isModeI16(imIn->mode);
    /* "I;16" and "I;16L" store little-endian samples, "I;16B" big-endian. */
    int swap = (imIn->mode == IMAGING_MODE_I_16B);
    void *lineOut = NULL;
    if (is16) {
        lineOut = calloc(imIn->xsize, sizeof(UINT16));
    } else {
        lineOut = calloc(imIn->xsize, sizeof(UINT32));
    }
    if (lineOut == NULL) {
        return ImagingError_MemoryError();
    }
    UINT16 *line16In = NULL;
    if (is16 && swap) {
        line16In = calloc(imIn->xsize, sizeof(UINT16));
        if (line16In == NULL) {
            free(lineOut);
            return ImagingError_MemoryError();
        }
    }

    // printf(">>> %d %d %d\n", radius, ww, fw);

    ImagingSectionEnter(&cookie);

    if (is16) {
        UINT16 *line16 = (UINT16 *)lineOut;
        for (y = 0; y < imIn->ysize; y++) {
            const UINT8 *rowIn = imIn->image8[y];
            UINT8 *rowOut = imOut->image8[y];
            int x;
            if (!swap) {
                if (imIn == imOut) {
                    ImagingLineBoxBlur16(
                        line16,
                        (const UINT16 *)rowIn,
                        imIn->xsize - 1,
                        radius,
                        edgeA,
                        edgeB,
                        ww,
                        fw
                    );
                    // Commit.
                    memcpy(rowOut, line16, imIn->xsize * sizeof(UINT16));
                } else {
                    ImagingLineBoxBlur16(
                        (UINT16 *)rowOut,
                        (const UINT16 *)rowIn,
                        imIn->xsize - 1,
                        radius,
                        edgeA,
                        edgeB,
                        ww,
                        fw
                    );
                }
            } else {
                for (x = 0; x < imIn->xsize; x++) {
                    line16In[x] = (UINT16)rowIn[x * 2] << 8 | rowIn[x * 2 + 1];
                }
                ImagingLineBoxBlur16(
                    line16,
                    line16In,
                    imIn->xsize - 1,
                    radius,
                    edgeA,
                    edgeB,
                    ww,
                    fw
                );
                for (x = 0; x < imIn->xsize; x++) {
                    rowOut[x * 2] = (UINT8)(line16[x] >> 8);
                    rowOut[x * 2 + 1] = (UINT8)line16[x];
                }
            }
        }
    } else if (imIn->image8) {
        for (y = 0; y < imIn->ysize; y++) {
            ImagingLineBoxBlur8(
                (imIn == imOut ? (UINT8 *)lineOut : imOut->image8[y]),
                imIn->image8[y],
                imIn->xsize - 1,
                radius,
                edgeA,
                edgeB,
                ww,
                fw
            );
            if (imIn == imOut) {
                // Commit.
                memcpy(imOut->image8[y], lineOut, imIn->xsize);
            }
        }
    } else {
        for (y = 0; y < imIn->ysize; y++) {
            ImagingLineBoxBlur32(
                imIn == imOut ? (pixel *)lineOut : (pixel *)imOut->image32[y],
                (pixel *)imIn->image32[y],
                imIn->xsize - 1,
                radius,
                edgeA,
                edgeB,
                ww,
                fw
            );
            if (imIn == imOut) {
                // Commit.
                memcpy(imOut->image32[y], lineOut, imIn->xsize * 4);
            }
        }
    }

    ImagingSectionLeave(&cookie);

    free(line16In);
    free(lineOut);

    return imOut;
}

Imaging
ImagingBoxBlur(Imaging imOut, Imaging imIn, float xradius, float yradius, int n) {
    int i;
    Imaging imTransposed;

    if (imOut->xsize == 0 || imOut->ysize == 0) {
        return imOut;
    }
    if (n < 1) {
        return ImagingError_ValueError("number of passes must be greater than zero");
    }
    if (xradius < 0 || yradius < 0) {
        return ImagingError_ValueError("radius must be >= 0");
    }

    if (imIn->mode != imOut->mode || imIn->type != imOut->type ||
        imIn->bands != imOut->bands || imIn->xsize != imOut->xsize ||
        imIn->ysize != imOut->ysize) {
        return ImagingError_Mismatch();
    }

    if (isModeI16(imIn->mode) &&
        (imIn->mode == IMAGING_MODE_I_16 || imIn->mode == IMAGING_MODE_I_16L ||
         imIn->mode == IMAGING_MODE_I_16B)) {
        /* Unsigned 16-bit grayscale, values in the 0..65535 range. */
        if (!isfinite(xradius) || !isfinite(yradius)) {
            return ImagingError_ValueError("radius must be finite");
        }
    } else {
        if (imIn->type != IMAGING_TYPE_UINT8) {
            return ImagingError_ModeError();
        }

        if (imIn->mode != IMAGING_MODE_RGB && imIn->mode != IMAGING_MODE_RGBA &&
            imIn->mode != IMAGING_MODE_RGBa && imIn->mode != IMAGING_MODE_RGBX &&
            imIn->mode != IMAGING_MODE_CMYK && imIn->mode != IMAGING_MODE_L &&
            imIn->mode != IMAGING_MODE_LA && imIn->mode != IMAGING_MODE_La) {
            return ImagingError_ModeError();
        }
    }

    /* Apply blur in one dimension.
       Use imOut as a destination at first pass,
       then use imOut as a source too. */

    if (xradius != 0) {
        ImagingHorizontalBoxBlur(imOut, imIn, xradius);
        for (i = 1; i < n; i++) {
            ImagingHorizontalBoxBlur(imOut, imOut, xradius);
        }
    }
    if (yradius != 0) {
        imTransposed = ImagingNewDirty(imIn->mode, imIn->ysize, imIn->xsize);
        if (!imTransposed) {
            return NULL;
        }

        /* Transpose result for blur in another direction. */
        ImagingTranspose(imTransposed, xradius == 0 ? imIn : imOut);

        /* Reuse imTransposed as a source and destination there. */
        for (i = 0; i < n; i++) {
            ImagingHorizontalBoxBlur(imTransposed, imTransposed, yradius);
        }
        /* Restore original orientation. */
        ImagingTranspose(imOut, imTransposed);

        ImagingDelete(imTransposed);
    }
    if (xradius == 0 && yradius == 0) {
        if (!ImagingCopy2(imOut, imIn)) {
            return NULL;
        }
    }

    return imOut;
}

static float
_gaussian_blur_radius(float radius, int passes) {
    float sigma2, L, l, a;

    sigma2 = radius * radius / passes;
    // from https://www.mia.uni-saarland.de/Publications/gwosdek-ssvm11.pdf
    // [7] Box length.
    L = sqrt(12.0 * sigma2 + 1.0);
    // [11] Integer part of box radius.
    l = floor((L - 1.0) / 2.0);
    // [14], [Fig. 2] Fractional part of box radius.
    a = (2 * l + 1) * (l * (l + 1) - 3 * sigma2);
    a /= 6 * (sigma2 - (l + 1) * (l + 1));

    return l + a;
}

Imaging
ImagingGaussianBlur(
    Imaging imOut, Imaging imIn, float xradius, float yradius, int passes
) {
    return ImagingBoxBlur(
        imOut,
        imIn,
        _gaussian_blur_radius(xradius, passes),
        _gaussian_blur_radius(yradius, passes),
        passes
    );
}
