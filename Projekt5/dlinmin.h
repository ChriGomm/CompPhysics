// #include "1dim_functions.h"
#define TOL 2.0e-4 //Tolerance passed to dbrent.

 //Global variables communicate with df1dim.
double *pcom,*xicom,(*nrfunc)(double []);
void (*nrdfun)(double [], double []);
int ncom;
void dlinmin(double p[], double xi[], int n, double *fret, double (*func)(double []),void (*dfunc)(double [], double []))
/*Given an n-dimensional point p[1..n] and an n-dimensional direction xi[1..n], moves and
resets p to where the function func(p) takes on a minimum along the direction xi from p,
and replaces xi by the actual vector displacement that p was moved. Also returns as fret
the value of func at the returned location p. This is actually all accomplished by calling the
routines mnbrak and dbrent.*/
{

double dbrent(double ax, double bx, double cx,double (*f)(double), double (*df)(double), double tol, double *xmin);
double f1dim(double x);
double df1dim(double x);
double (*p_f1d)(double);
double (*p_df1d)(double);
p_df1d = df1dim;
p_f1d= f1dim;
void mnbrak(double *ax, double *bx, double *cx, double *fa, double *fb,double *fc, double (*func)(double));
int j;
double xx,xmin,fx,fb,fa,bx,ax;
ncom=n; //Define the global variables.
double vec1[5];
double vec2[5];
pcom=vec1;
xicom=vec2;
nrfunc=func;
nrdfun=dfunc;
for (j=0;j<n;j++) {
pcom[j]=p[j];
xicom[j]=xi[j];
}
ax=0.0; //Initial guess for brackets.
xx=1.0;
mnbrak(&ax,&xx,&bx,&fa,&fx,&fb,p_f1d);
*fret=dbrent(ax,xx,bx,p_f1d,p_df1d,TOL,&xmin);
// printf("xmin:%e\n",xmin);
// printf("%f, %f\n",xi[0],xi[1]);
// printf("%f, %f\n",p[0],p[1]);
for (j=0;j<n;j++) { //Construct the vector results to return.
xi[j] *= xmin;
p[j] += xi[j];
}
// free_vector(xicom,1,n);
// free_vector(pcom,1,n);
}

double df1dim(double x)
{
int j;
double df1=0.0;
double xt[ncom];
double df[ncom];
for (j=0;j<ncom;j++) xt[j]=pcom[j]+x*xicom[j];
(*nrdfun)(xt,df);
for (j=0;j<ncom;j++) df1 += df[j]*xicom[j];
// free_vector(df,1,ncom);
// free_vector(xt,1,ncom);
return df1;
}
double f1dim(double x)
{
int j;
double f1=0.0;
double xt[ncom];
for (j=0;j<ncom;j++) 
xt[j]=pcom[j]+x*xicom[j];

f1 = (*nrfunc)(xt);

return f1;
}