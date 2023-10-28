#include <stdio.h>
// #include "nrutil.h"
#define ITMAX 200
#define EPS 1.0e-10
#include "dbrent.h"
#include "mnbrak.h"
#include "dlinmin.h"

const int ndim = 5;
int data[120][2];
double dev[120];
double point[ndim];
void derv_ff(double v [],double storage []);
double fit_func(double v[ndim]);
double test_f(double vi[ndim]);
void test_g(double vi[ndim],double store[ndim]);
void d_func1(double x[2],double grad[2]);
double func1(double x[2]);
void der_test2(double x[2], double grad[2]);
double test2(double x[2]);
/*Here ITMAX is the maximum allowed number of iterations, while EPS is a small number to
rectify the special case of converging to exactly zero function value.
#define FREEALL free_vector(xi,1,n);free_vector(h,1,n);free_vector(g,1,n);*/
void frprmn(double p[], int n, double ftol, int *iter, double *fret, double (*func)(double []), void (*dfunc)(double [], double []))
/*Given a starting point p[1..n], Fletcher-Reeves-Polak-Ribiere minimization is performed on a
function func, using its gradient as calculated by a routine dfunc. The convergence tolerance
on the function value is input as ftol. Returned quantities are p (the location of the minimum),
iter (the number of iterations that were performed), and fret (the minimum value of the
function). The routine linmin is called to perform line minimizations.*/
{
void dlinmin(double p[], double xi[], int n, double *fret, double (*func)(double []),void (*dfunc)(double [], double []));
int j,its;
double gg,gam,fp,dgg;


double g[n];
double h[n];
double xi[n];
fp=(*func)(p); //Initializations.
(*dfunc)(p,xi);
for (j=0;j<n;j++) {
g[j] = -xi[j];
xi[j]=h[j]=g[j];
double store_p[n];
}
double rtol;
for (its=1;its<=ITMAX;its++) { //Loop over iterations.
*iter=its;
printf("p: %f, %.8f\n",p[0],p[1]);
printf("xi2: %f, %.8f\n",xi[0],xi[1]);
dlinmin(p,xi,n,fret,func,dfunc);
printf("fret: %.10f\n",*fret); //Next statement is the normal return:
if ((rtol=2.0*fabs(*fret-fp)) <= ftol*(fabs(*fret)+fabs(fp)+EPS)) {
//FREEALL
printf("rtol:%e\n",rtol);
return ;
}
fp= *fret;
(*dfunc)(p,xi);
printf("xi: %f, %.8f\n",xi[0],xi[1]);
dgg=gg=0.0;
for (j=0;j<n;j++) {
    gg += g[j]*g[j];
dgg += xi[j]*xi[j];  //This statement for Fletcher-Reeves.
// dgg += (xi[j]+g[j])*xi[j]; //This statement for Polak-Ribiere.
}
if (gg == 0.0) { //Unlikely. If gradient is exactly zero then
//FREEALL           we are already done.
printf("gg is 0.");
return;
}
gam=dgg/gg;
for (j=0;j<n;j++) {
g[j] = -xi[j];
xi[j]=h[j]=g[j]+gam*h[j];
}
}
printf("Too many iterations in frprmn");
}


int main()
{
    FILE *datei;
    datei = fopen("Agdecay.dat","r");
    
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
    // for ( i=0;i<5;++i)
    // {
    //     printf("%i\t",data[i][0]);
    //     printf("%i\n",data[i][1]);
    // }
    for (i=0;i<120;++i)
    dev[i] = sqrt((double)data[i][1]);
    // for (i =0;i<5;++i)
    // printf("%i\n",data[i][1]);
    point[1]= log(2)/(3600*24*41);
    point[3]=log(2)/(3600*24*462.6);
    point[0]=1000;
    point[2]=1000;
    point[4]=0.5;
    // point[1]= 0.043245;
    // point[3]=9;
    // point[0]=870.311265*4;
    // point[2]=1.913512;
    // point[4]=7.978376;
    // point[1]= 2;
    // point[0]=50;
    // point[0]=-30;
    // point[2]=190;
    // point[4]=-25;
    int iter_count=0;
    double fret=0;
    double (*ff)(double[]);
    ff = fit_func;
    void (*dff)(double[],double[]);
    dff = derv_ff;
    frprmn(point, 5, 10e-15, &iter_count, &fret, fit_func, derv_ff);
    printf("number of iterations: %i\n",iter_count);
    for (int i =0;i<ndim;++i)
    {
        printf("%f\t",point[i]);
    }
    printf("\n minimum value:%f\n",fret);
    return 0;


}
double fit_func(double v[ndim])
{
        double result =0;
        for (int i=0;i<120;++i)
        result += (v[0]*exp(-v[1]*(double)i*5)*(1-exp(-v[1]*5))+v[2]*exp(-v[3]*(double)i*5)*(1-exp(-v[3]*5))+v[4]-(double)data[i][1])*(v[0]*exp(-v[1]*(double)i*5)*(1-exp(-v[1]*5))+v[2]*exp(-v[3]*(double)i*5)*(1-exp(-v[3]*5))+v[4]-(double)data[i][1])/dev[i]/dev[i];
        if (v[1]>0.5 || v[3]>0.5)
        result += exp(30);
        return result;
}
void derv_ff(double v [],double storage [])
{
    for (int i=0;i<5;++i)
    storage[i]=0.0;
    for (int i=0;i<120;++i)
    {
        storage[0]=(v[0]*exp(-v[1]*(double)i*5)*(1-exp(-v[1]*5))+v[2]*exp(-v[3]*(double)i*5)*(1-exp(-v[3]*5))+v[4]-(double)data[i][1])*exp(-v[1]*(double)i*5)*(1-exp(-v[1]*5))*2/dev[i]/dev[i];
        storage[1]=(v[0]*exp(-v[1]*(double)i*5)*(1-exp(-v[1]*5))+v[2]*exp(-v[3]*(double)i*5)*(1-exp(-v[3]*5))+v[4]-(double)data[i][1])*v[0]*exp(-v[1]*(double)i*5)*(-(double)i*5+5*(double)(i+1)*exp(-v[1]*5))*2/dev[i]/dev[i];
        storage[2]=(v[0]*exp(-v[1]*(double)i*5)*(1-exp(-v[1]*5))+v[2]*exp(-v[3]*(double)i*5)*(1-exp(-v[3]*5))+v[4]-(double)data[i][1])*exp(-v[3]*(double)i*5)*(1-exp(-v[3]*5))*2/dev[i]/dev[i];
        storage[3]=(v[0]*exp(-v[1]*(double)i*5)*(1-exp(-v[1]*5))+v[2]*exp(-v[3]*(double)i*5)*(1-exp(-v[3]*5))+v[4]-(double)data[i][1])*v[2]*exp(-v[3]*(double)i*5)*(-(double)i*5+5*(double)(i+1)*exp(-v[3]*5))*2/dev[i]/dev[i];
        storage[4]=(v[0]*exp(-v[1]*(double)i*5)*(1-exp(-v[1]*5))+v[2]*exp(-v[3]*(double)i*5)*(1-exp(-v[3]*5))+v[4]-(double)data[i][1])*2/dev[i]/dev[i];
}  
}    

double test_f(double vi[ndim])
{
    double result =0;
        for (int i=0;i<120;++i)
        result += (vi[0]*exp(-vi[1]*(double)i*5)+vi[2]*vi[2]+vi[3]*vi[3]+vi[4]*vi[4]-(double)data[i][1])*(vi[0]*exp(-vi[1]*(double)i*5)+vi[2]*vi[2]+vi[3]*vi[3]+vi[4]*vi[4]-(double)data[i][1])/dev[i]/dev[i];
        return result;

}
void test_g(double vi[ndim],double store[ndim])
{
    for (int i=0;i<5;++i)
    store[i]=0.0;
    for (int i=0;i<120;++i)
    {
        store[0]+=(vi[0]*exp(-vi[1]*(double)i*5)+vi[2]*vi[2]+vi[3]*vi[3]+vi[4]*vi[4]-(double)data[i][1])*exp(-vi[1]*(double)i*5)*2/dev[i]/dev[i];
        store[1]+=(vi[0]*exp(-vi[1]*(double)i*5)+vi[2]*vi[2]+vi[3]*vi[3]+vi[4]*vi[4]-(double)data[i][1])*vi[0]*(double)(-5*i)*2/dev[i]/dev[i];
        store[2]+=(vi[0]*exp(-vi[1]*(double)i*5)+vi[2]*vi[2]+vi[3]*vi[3]+vi[4]*vi[4]-(double)data[i][1])*4*vi[2]/dev[i]/dev[i];
        store[3]+=(vi[0]*exp(-vi[1]*(double)i*5)+vi[2]*vi[2]+vi[3]*vi[3]+vi[4]*vi[4]-(double)data[i][1])*4*vi[3]/dev[i]/dev[i];
        store[4]+=(vi[0]*exp(-vi[1]*(double)i*5)+vi[2]*vi[2]+vi[3]*vi[3]+vi[4]*vi[4]-(double)data[i][1])*4*vi[4]/dev[i]/dev[i];
    }
}
double func1(double x[2])
{
  return x[0]*x[0]*x[0]*x[0]-x[1]*x[0]*x[0]-3*x[0]*x[1]*x[1]+x[1]*x[1]*x[1]*x[1];
}

void d_func1(double x[2],double grad[2])
{
  grad[0] = 4*x[0]*x[0]*x[0]-2*x[1]*x[0]-3*x[1]*x[1];
  grad[1]= -x[0]*x[0]-6*x[1]*x[0]+4*x[1]*x[1]*x[1];
}
double test2(double x[2])
{
    return x[0]*x[0]*3+x[1]*x[1]*2;
}
void der_test2(double x[2], double grad[2])
{
    grad[0]=6*x[0];
    grad[1]=4*x[1];
    
}