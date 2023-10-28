#include <math.h>
#include <stdio.h>
#define TINY 1.0e-10 //A small number.
#define NMAX 500000 //Maximum allowed number of function evaluations.
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
    // psum[1]= log(2)/(3600*24*41);
    // psum[3]=log(2)/(3600*24*462.6);
    // psum[0]=1000;
    // psum[2]=1;
    // psum[4]=0.5;
    psum[1]= 0.043245;
    psum[3]=9;
    psum[0]=870.311265*4;
    psum[2]=1.913512;
    psum[4]=7.978376;
    
    for ( int j=0;j<=ndim;++j)
    {
        for (int k=0;k<ndim;++k)
        {
            p[j][k] = psum[k];
        }
        if (j!=ndim)
        {
            p[j][j]+= 10;
        }
        else
        p[j][0] -=5000;
    }
    double y[ndim+1];
    for (i=0;i<=ndim;++i)
    {
        vec(i);
        y[i]=fit_func(vector);
        // printf("%f\n",y[i]);
    }
    double ftol = 1.0e-12;
    int nfunk =0;
    double (*func)(double v[ndim]);
    amoeba(y, ftol, &nfunk);
    for (i=0;i<ndim;++i)
    printf("%f\n",psum[i]);
    printf("%i\n",nfunk);
    printf("hihgest: %f\n",fit_func(vector));

    return 0;
}

void vec(int line)
{
    for (int k=0;k<ndim;++k)
    vector[k]=p[line][k];

}

double fit_func(double v[ndim])
{
        double result =0;
        for (int i=0;i<120;++i)
        result += (v[0]*exp(-v[1]*(double)i*5)*(1-exp(-v[1]*5))+v[2]*exp(-v[3]*(double)i*5)*(1-exp(-v[3]*5))+v[4]-(double)data[i][1])*(v[0]*exp(-v[1]*(double)i*5)*(1-exp(-v[1]*5))+v[2]*exp(-v[3]*(double)i*5)*(1-exp(-v[3]*5))+v[4]-(double)data[i][1])/dev[i]/dev[i];
        return result;
}
void get_psum() 
{
    for (int j=0;j<ndim;j++) 
{
double sum =0;
for (int i=0;i<(ndim+1);i++) 
{

sum += p[i][j];
}
psum[j]=sum/(ndim+1);
}
}

void get_plane(int ihi) 
{
    for (int j=0;j<ndim;j++) 
{
double sum =0;
for (int i=0;i<(ndim+1);i++) 
{
if (i!=ihi)
sum += p[i][j];
}
pplane[j]=sum/(ndim);
}
}

void SWAP(double a,double b) 
{
double swap=(a);
(a)=(b);
(b)=swap;
}

// Extrapolates by a factor fac through the face of the simplex across from the high point, tries
// it, and replaces the high point if the new point is better.
double amotry(double y[ndim], int ihi, double fac)
{
int j;
double fac1,fac2,ytry;
double ptry[ndim];
// fac1=(1.0-fac)/ndim;
// fac2=fac1-fac;
get_plane(ihi);
for (j=0;j<ndim;j++) 
ptry[j]=-(pplane[j]-p[ihi][j])*fac+pplane[j];
ytry= fit_func(ptry); //Evaluate the function at the trial point.
if (ytry < y[ihi]) { //If it’s better than the highest, then replace the highest.
y[ihi]=ytry;
for (j=0;j<ndim;j++) {
psum[j] += (ptry[j]-p[ihi][j])/ndim;
p[ihi][j]=ptry[j];
}
}

return ytry;
}
/*Multidimensional minimization of the function funk(x) where x[1..ndim] is a vector in ndim
dimensions, by the downhill simplex method of Nelder and Mead. The matrix p[1..ndim+1]
[1..ndim] is input. Its ndim+1 rows are ndim-dimensional vectors which are the vertices of
the starting simplex. Also input is the vector y[1..ndim+1], whose components must be preinitialized to the values of funk evaluated at the ndim+1 vertices (rows) of p; and ftol the
fractional convergence tolerance to be achieved in the function value (n.b.!). On output, p and
y will have been reset to ndim+1 new points all within ftol of a minimum function value, and
nfunk gives the number of function evaluations taken.*/
void amoeba(double y[ndim],double ftol,int *nfunk)
{


int i,ihi,ilo,inhi,j,mpts=ndim+1;
double rtol,sum,swap,ysave,ytry;


*nfunk=0;
get_psum();

for (;;) 
{
ilo=1;
/*First we must determine which point is the highest (worst), next-highest, and lowest
(best), by looping over the points in the simplex.*/
ihi = y[1]>y[2] ? (inhi=2,1) : (inhi=1,2);
for (i=0;i<mpts;i++) 
{
if (y[i] <= y[ilo]) 
ilo=i;
if (y[i] > y[ihi]) 
{
inhi=ihi;
ihi=i;
} 
}
for (i=0;i<mpts;i++) 
{
if (y[i] > y[inhi] && i != ihi) 
inhi=i;
}

rtol=2.0*fabs(y[ihi]-y[ilo]);//(fabs(y[ihi])+fabs(y[ilo])+TINY);
/*Compute the fractional range from highest to lowest and return if satisfactory.*/
if (rtol < ftol) { //If returning, put best point and value in slot 1.
SWAP(y[1],y[ilo]);
for (i=0;i<ndim;i++) 
SWAP(p[1][i],p[ilo][i]);
break;
}
if (*nfunk >= NMAX) 
{
    printf("NMAX exceeded");
    break;
}
*nfunk += 2;
// Begin a new iteration. First extrapolate by a factor −1 through the face of the simplex
// across from the high point, i.e., reflect the simplex from the high point.
ytry=amotry(y,ihi,-1.0);
if (ytry <= y[ilo])
// Gives a result better than the best point, so try an additional extrapolation by a
// factor 2.
ytry=amotry(y,ihi,2.0);
else if (ytry >= y[inhi]) {
// The reflected point is worse than the second-highest, so look for an intermediate
// lower point, i.e., do a one-dimensional contraction.
ysave=y[ihi];
ytry=amotry(y,ihi,0.5);
if (ytry >= ysave) { //Can’t seem to get rid of that high point. Better
for (i=0;i<mpts;i++) { //contract around the lowest (best) point.

if (i != ilo) {
for (j=0;j<ndim;j++)
p[i][j]=psum[j]=0.5*(p[i][j]+p[ilo][j]);
y[i]=fit_func(psum);
}
}
*nfunk += ndim; //Keep track of function evaluations.
get_psum(); //Recompute psum.
}
} else --(*nfunk); //Correct the evaluation count.
} //Go back for the test of doneness and the next iteration.
for (int k=0;k<ndim;++k)
{
    if (k==ndim-1)
    printf("%f\n",p[ilo][k]); 
    else
    printf("%f\t",p[ilo][k]);
}
vec(1);
printf("lowest: %f\n",fit_func(vector));
vec(ihi);
printf("%e\n",rtol);
}


