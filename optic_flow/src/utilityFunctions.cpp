
#include "../include/optic_flow/utilityFunctions.h"

void rotate2d(double &x, double &y, double alpha)
{
    double cs = cos(alpha);// cos and sin of rot maxtrix
    double sn = sin(alpha);

    double x_n = x*cs-y*sn;
    double y_n = x*sn+y*cs;
    x = x_n;
    y = y_n;


}

