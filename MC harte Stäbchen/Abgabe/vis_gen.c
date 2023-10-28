#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
// Dieses Programm generiert eine Gleichgewichtsverteilung für gegebenes z
const int M =64;
const int L = 8;
int v_numb;
int h_numb;
double max = (double)RAND_MAX;
void del_occ(int lat[M][M], int x_pos, int y_pos,int a);
void ins_occ(int lat[M][M], int x_pos, int y_pos,int a);
int occ_check(int lat[M][M], int ps[2],int a);
void sweep(int p_v[M*M/L][2],int p_h[M*M/L][2],int occ_lat[M][M], double zet, int *ho_numb,int *ver_numb,int *proto);

int main()
{
    srandom(time(0));
    // initializing
    time_t t1 = time(0);
    
    // ** z hier eingeben **
    double z =  0.56;
    

    FILE *datei;
    datei = fopen("vis_seed.txt","w");
    
    int v_particles[M*M/L][2]={0};
    int h_particles[M*M/L][2]={0};
    int lat[M][M]={0};
    v_numb = 0;
    h_numb = 0;
    int protocoll = 0;
    for (unsigned long n=0;n<(202000100);++n)
    {
    
        sweep(v_particles,h_particles,lat,z,&h_numb,&v_numb,&protocoll);
        
        
    }
// Gitter wird abgespeichert
for ( int i=0;i<M;++i)
{
    for (int j=0;j<M;++j)
    {
        if (j==M-1)
        fprintf(datei,"%i\n",lat[i][j]);
        else
        fprintf(datei,"%i\t",lat[i][j]);

    }
}
// Teilchenpositionen werden abgespeichert
// vertikal
for ( int i=0;i<M*M/L;++i)
{
    for (int j=0;j<2;++j)
    {
        if (j==1)
        fprintf(datei,"%i\n",v_particles[i][j]);
        else
        fprintf(datei,"%i\t",v_particles[i][j]);

    }
}
// horizontal
for ( int i=0;i<M*M/L;++i)
{
    for (int j=0;j<2;++j)
    {
        if (j==1)
        fprintf(datei,"%i\n",h_particles[i][j]);
        else
        fprintf(datei,"%i\t",h_particles[i][j]);

    }
}
// in der Letzten Zeile werden die Teilchenzahlen und z abgelegt
fprintf(datei,"%i\t",v_numb);
fprintf(datei,"%i\t",h_numb);
fprintf(datei,"%f",z);



fclose(datei);




time_t t2 = time(0);
printf("%ld",t2-t1);
    
    
    return 0;
}

void del_occ(int lat[M][M], int x_pos, int y_pos,int a)
{
    if (a==0)
    {
    for ( int i=0;i<L;++i)
    {
            
        lat[(x_pos+i)%M][y_pos]=0;
    }
    }
    if ( a==1)
    {
        for ( int i=0;i<L;++i)
    {
            
        lat[x_pos][(y_pos+i)%M]=0;
    }
    }

    
}
void ins_occ(int lat[M][M], int x_pos, int y_pos,int a)
{
    if (a==0)
    {
    for ( int i=0;i<L;++i)
    {
            
        lat[(x_pos+i)%M][y_pos]=-1;
    }
    }
    if ( a==1)
    {
        for ( int i=0;i<L;++i)
    {
            
        lat[x_pos][(y_pos+i)%M]=1;
    }
    }

    
}
int occ_check(int lat[M][M], int ps[2],int a)
{
    int result =0;
    if (a==0)
    {
    for ( int i=0;i<L;++i)
    {
            
        result += lat[(ps[0]+i)%M][ps[1]]*lat[(ps[0]+i)%M][ps[1]];
    }
    }
    else if ( a==1)
    {
        for ( int i=0;i<L;++i)
    {
            
        result += lat[ps[0]][(ps[1]+i)%M]*lat[ps[0]][(ps[1]+i)%M];
    }
    }
    return result;
}
void sweep(int p_v[M*M/L][2],int p_h[M*M/L][2],int occ_lat[M][M], double zet,int *ho_numb,int *ver_numb,int *proto)
{
    
    if (random()%2==0)

    {
        
        
        if (((double)random()/max)<((double)((*ver_numb)+(*ho_numb))/(2*(double)(M*M)*zet)))
        {
        *proto += 1;
        int del_part = (int)((double)random()/max*(double)((*ver_numb)+(*ho_numb)));
        if (del_part<(*ho_numb))
        {
            if (*ho_numb>0)
            {
            
            del_occ(occ_lat,p_h[del_part][0],p_h[del_part][1],1);
            *ho_numb -= 1;
            for ( int i=del_part;i<*ho_numb;++i)
            {
                p_h[i][0]=p_h[i+1][0];
                p_h[i][1]=p_h[i+1][1];
            }
            }
        }
        else
        {
            if (*ver_numb>0)
            {
            del_occ(occ_lat,p_v[del_part-(*ho_numb)][0],p_v[del_part-(*ho_numb)][1],0);
            *ver_numb -= 1;
            for ( int i=del_part-(*ho_numb);i<*ver_numb;++i)
            {
                p_v[i][0]=p_v[i+1][0];
                p_v[i][1]=p_v[i+1][1];
            }
            }
        }
        }
        

    }
    else
    {
        int ins_pos = (int)((double)random()/max*(M*M));  
        int pos[2] = {(ins_pos-ins_pos%M)/M,ins_pos%M};
        if (random()%2==0)
        {
            if (occ_check(occ_lat,pos,1)==0)
            {
                
                // muss die alpha Übergangsrate nicht auswerten, da sie immer 1 ist.
                p_h[*ho_numb][0] = pos[0];
                p_h[*ho_numb][1] = pos[1];
                ins_occ(occ_lat,pos[0],pos[1],1);
                *ho_numb += 1;

            }
        }
        else
        {
            if (occ_check(occ_lat,pos,0)==0)
            {
                // muss die alpha Übergangsrate nicht auswerten, da sie immer 1 ist.
                p_v[*ver_numb][0] = pos[0];
                p_v[*ver_numb][1] = pos[1];
                ins_occ(occ_lat,pos[0],pos[1],0);
                *ver_numb += 1;

            }
        }
    }
}
