#if !defined(MSVC_9_FMAX) && defined(_MSC_VER) && _MSC_VER <= 1600
#define MSVC_9_FMAX

#include <float.h>

#define isnan(x) _isnan((x))

static int signbit(double x)
{
    return x > 0;
}

static double fmax(double x, double y)
{
    /* z > nan for z != nan is required by C the standard */
    int xnan = isnan(x), ynan = isnan(y);

    if (xnan || ynan)
    {
        if (xnan && !ynan)
            return y;
        if (!xnan && ynan)
            return x;
        return x;
    }

    /* +0 > -0 is preferred by C the standard */
    if (x == 0 && y == 0)
    {
        int xs = signbit(x), ys = signbit(y);
        if (xs && !ys)
            return y;
        if (!xs && ys)
            return x;
        return x;
    }

    if (x > y) {
        return x;
    }
    else {
        return y;
    }
}
#endif
