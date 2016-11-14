
void rotate2d(double &x, double &y, double alpha)
{
    double cs = cos(phi);// cos and sin of rot maxtrix
    double sn = sin(phi);

    double x_n = x*cs-y*sn;
    double y_n = x*sn+y*cs;
    x = vxm_n;
    y = vym_n;


}
