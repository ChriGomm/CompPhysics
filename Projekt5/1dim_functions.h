extern int ncom; //Defined in dlinmin.
extern float *pcom,*xicom,(*nrfunc)(float []);
extern void (*nrdfun)(float [], float []);
float df1dim(float x)
{
int j;
float df1=0.0;
float *xt,*df;
float vec3[ncom];
float vec4[ncom];
xt=vec3;
df=vec4;
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
float *xt;
float vec3[ncom];
float vec4[ncom];
xt=vec3;
for (j=0;j<ncom;j++) xt[j]=pcom[j]+x*xicom[j];
f1 = (*nrfunc)(xt);

return f1;
}