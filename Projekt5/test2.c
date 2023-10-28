#include <stdio.h>
#include <math.h>
#include "dbrent.h"
#include "mnbrak.h"
#define TOL 2.0e-4 //Tolerance passed to dbrent.

 //Global variables communicate with df1dim.
float *pcom,*xicom,(*nrfunc)(float []);
void (*nrdfun)(float [], float []);
int ncom;
/*Given an n-dimensional point p[1..n] and an n-dimensional direction xi[1..n], moves and
resets p to where the function func(p) takes on a minimum along the direction xi from p,
and replaces xi by the actual vector displacement that p was moved. Also returns as fret
the value of func at the returned location p. This is actually all accomplished by calling the
routines mnbrak and dbrent.*/
void dlinmin(float p[], float xi[], int n, float *fret, float (*func)(float []),void (*dfunc)(float [], float []))
{

float dbrent(float ax, float bx, float cx,float (*f)(float), float (*df)(float), float tol, float *xmin);
float f1dim(float x);
float df1dim(float x);
float (*p_f1d)(float);
float (*p_df1d)(float);
p_df1d = df1dim;
p_f1d= f1dim;
void mnbrak(float *ax, float *bx, float *cx, float *fa, float *fb,float *fc, float (*func)(float));
int j;
float xx,xmin,fx,fb,fa,bx,ax;
ncom=n; //Define the global variables.
float vec1[5];
float vec2[5];
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
printf("xmin:%e\n",xmin);


for (j=0;j<n;j++) { //Construct the vector results to return.
xi[j] *= xmin;
p[j] += xi[j];
}
printf("%f, %f\n",xi[0],xi[1]);
printf("%f, %f\n",p[0],p[1]);
// free_vector(xicom,1,n);
// free_vector(pcom,1,n);
}

float df1dim(float x)
{
int j;
float df1=0.0;
float xt[ncom];
float df[ncom];
for (j=0;j<ncom;j++) xt[j]=pcom[j]+x*xicom[j];
(*nrdfun)(xt,df);
for (j=0;j<ncom;j++) df1 += df[j]*xicom[j];
// free_vector(df,1,ncom);
// free_vector(xt,1,ncom);
return df1;
}
float f1dim(float x)
{
int j;
float f1=0.0;
float xt[ncom];
for (j=0;j<ncom;j++) 
xt[j]=pcom[j]+x*xicom[j];

f1 = (*nrfunc)(xt);

return f1;
}

float func1(float x[2]);
void d_func1(float x[2], float grad[2]);

int main()
{

  float point[2] = {3,2};
  float dirc[2]= {1,0};
  float f_val=0;
  dlinmin(point,dirc,2,&f_val,func1,d_func1);
    // float ax = 4;
    // float bx= 3.8;
    // float cx = 4.15;
    // float fa = func(ax);
    // float fb = func(bx);
    // float fc = func(cx);
    // float x_0;
    // mnbrak(&ax,&bx,&cx,&fa,&fb,&fc,func);

    // printf("ax:%f,bx:%f,cx:%f,fa:%f,fb:%f,fc:%f\n",ax,bx,cx,fa,fb,fc);
    // dbrent(ax,bx,cx,func,d_func,1e-5,&x_0);
    printf("min = %f,%f\n",point[0],point[1]);

  return 0;
}

float func1(float x[2])
{
  return x[0]*x[0]*x[0]-x[1]*x[0]*x[0]-3*x[0]*x[1]*x[1];
}

void d_func1(float x[2],float grad[2])
{
  grad[0] = 3*x[0]*x[0]-2*x[1]*x[0]-3*x[1]*x[1];
  grad[1]= -x[0]*x[0]-6*x[1]*x[0];
}