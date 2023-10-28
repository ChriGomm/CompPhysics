#include <stdio.h>
#include <math.h>

const int ndim=5;
double psum[ndim];
double pplane[ndim];
double p[ndim+1][ndim];
double ptry[ndim];
int data[120][2]={0};
double dev[120];
double vector[ndim];
double amotry(double y[ndim], int ihi, double fac);
double fit_func(double v[ndim]);
void get_psum();
void get_plane(int ihi);
void SWAP(double a,double b);
void amoeba(double y[ndim], double ftol, int *nfunk);
void vec(int line);
double fit_func(double v[ndim]);

int main() {

    
    FILE *datei;
    datei = fopen("Agdecay.dat","r");
    int data[120][2];
    int store[2];
    int i =0;
    while (!feof(datei))
    {
        fscanf(datei,"%i\t%i",&store[0],&store[1]);
        data[i][0]=store[0];
        data[i][1] =store[1];
        ++i;
    }
    fclose(datei);
    for (i=0;i<120;++i)
    dev[i] = sqrt((double)data[i][1]);
    // datei = fopen("ag_decay.txt","w");
    // for ( int i=0;i<120;++i){

    
    // fprintf(datei,"%i\t",data[i][0]);
    // fprintf(datei,"%i\n",data[i][1]);}
    // fclose(datei);
    double tp[ndim]= {1,2,3,4,5};
    printf("%f\n",fit_func(tp));

    return 0;
}

double fit_func(double v[ndim])
{
        double result =0;
        for (int i=0;i<120;++i)
        result += (v[0]*exp(-v[1]*(double)i*5)*(1-exp(-v[1]*5))+v[2]*exp(-v[3]*(double)i*5)*(1-exp(-v[3]*5))+v[4]-(double)data[i][1])*(v[0]*exp(-v[1]*(double)i*5)*(1-exp(-v[1]*5))+v[2]*exp(-v[3]*(double)i*5)*(1-exp(-v[3]*5))+v[4]-(double)data[i][1])/dev[i]/dev[i];
        return result;
}