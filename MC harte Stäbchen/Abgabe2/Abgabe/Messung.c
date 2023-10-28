#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
// Gittergröße
const int M =64;
// Stäbchenlänge
const int L = 8;
// Anzahl vertikale Stäbchen
int v_numb;
// Anzahl horizontale Stäbchen
int h_numb;
// maximale Zahl, die die random Funktion ausgibt.
double max = (double)RAND_MAX;

void del_occ(int lat[M][M], int x_pos, int y_pos,int a);
void ins_occ(int lat[M][M], int x_pos, int y_pos,int a);
int occ_check(int lat[M][M], int ps[2],int a);
void sweep(int p_v[M*M/L][2],int p_h[M*M/L][2],int occ_lat[M][M], double zet, int *ho_numb,int *ver_numb,int *proto);

int main()
{
    // Zufallsfunktion initialisieren
    srandom(time(0));
    
    // Startpunkt der Zeitstoppung
    time_t t1 = time(0);
    
    // Aktivität z
    double z[3] = { 0.56, 0.84 ,1.1};
    
    // Speicherdokument öffnen
    FILE *datei;
    datei = fopen("test7.txt","w");

    // Schleife der z
    for (int idx=0;idx<2;++idx)
    {
    // Liste mit Positionen wird initialisiert für beide Orientierungen extra ausgelegt.
    int v_particles[M*M/L][2]={0};
    int h_particles[M*M/L][2]={0};
    // Gitter zur Überprüfung der Belegung wird initialisiert
    int lat[M][M]={0};
    v_numb = 0;
    h_numb = 0;
    // hier wird protokolliert, wie oft ein Stäbchen gelöscht wird ( habe ich zum Debugging verwendet)
    int protocoll = 0;

    // Schleife der Iterationen
    for (unsigned long n=0;n<(22000000000+1);++n)
    {
    
        sweep(v_particles,h_particles,lat,z[idx],&h_numb,&v_numb,&protocoll);
        
        // gemessen wird jeweils nach Verstreichen der Thermalisierungszeit
        if (n%(20000)==0 & n!=0 )
        {
        fprintf(datei,"%i\t",h_numb);
        fprintf(datei,"%i\t",v_numb);
        fprintf(datei,"%i\n",protocoll);
        protocoll =0;
        }
        
        
        
        
        
    }
    }
fclose(datei);



// es wird zum Schluss ausgegeben wie lange das Programm gebraucht hat
time_t t2 = time(0);
printf("%ld",t2-t1);
    
    
    return 0;
}

// Löschen eines Teilchens aus dem Gitter. Parameter a gibt an ob ein horizontales (1) oder vertikales (0) Stäbchen gelöscht wird
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
// Teilchen wird ins Gitter eingefügt
// im Gitter steht eine 0 für einen freien Platz, -1 für einen von einem vertikalen Stäbchen belegten Platz, und 1 signalisiert entsprechend ein 
//  horizontales Stäbchen
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
// es wird überprüft, ob an einer Position ein Stäbchen eingefügt werden kann. a=1 -> h und a=0 -> v
// wenn ein teilchen an dieser Stelle eingefügt werden kann, wird eine 0 zurück gegeben, sonst eine andere Zahl
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

// Funktion für das Monte-Calo Update
void sweep(int p_v[M*M/L][2],int p_h[M*M/L][2],int occ_lat[M][M], double zet,int *ho_numb,int *ver_numb,int *proto)
{
    // zuerst wird gelost, ob ein Teilchen gelöscht (True) oder eingefügt (False) werden soll
    if (random()%2==0)

    {
        
        // Es wird gelost, ob ein Teilchen gelöscht werden kann
        if (((double)random()/max)<((double)((*ver_numb)+(*ho_numb))/(2*(double)(M*M)*zet)))
        {
        // wenn ja wird dies im Protokoll vermerkt
        *proto += 1;
        // Nummer des Zustands des gelöschten Teilchens wird gelost (0 - (N-1)) N Gesamtzahl der Stäbchen
        int del_part = (int)((double)random()/max*(double)((*ver_numb)+(*ho_numb)));
        // wenn die Nummer einem horizontalen Zustand entspricht wird hier fortgefahren
        if (del_part<(*ho_numb))
        {
            // nur löschen wenn es auch Teilchen gibt
            if (*ho_numb>0)
            {
            
            del_occ(occ_lat,p_h[del_part][0],p_h[del_part][1],1);
            *ho_numb -= 1;
            // der entsprechende Zustand wird aus den Büchern entfernt und die Lücke geschlossen
            for ( int i=del_part;i<*ho_numb;++i)
            {
                p_h[i][0]=p_h[i+1][0];
                p_h[i][1]=p_h[i+1][1];
            }
            }
        }
        else
        {   
            // für verikale Zustände das gleiche
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
    // es wurde gelost ein Teilchen einzufügen
    else
    {   
        // einzufügende Position von 1, .., M^2 um eine auswertung der random Funktion zu sparen.
        int ins_pos = (int)((double)random()/max*(M*M));  
        // Daraus wird dann eine verwendbare Position gemacht
        int pos[2] = {(ins_pos-ins_pos%M)/M,ins_pos%M};
        // es wird gelost, ob ein horizontales (True) oder vertikales (False) Stäbchen eingefügt wird
        if (random()%2==0)
        {
            // ist die Position belegt?
            if (occ_check(occ_lat,pos,1)==0)
            {
                
                // muss die alpha Übergangsrate nicht auswerten, da sie immer 1 ist. Habe ich überprüft (nur bei z=0.05, möglich, tritt dort
                //  aber auch nicht auf)
                // Stäbchen wird hinzugefügt
                p_h[*ho_numb][0] = pos[0];
                p_h[*ho_numb][1] = pos[1];
                ins_occ(occ_lat,pos[0],pos[1],1);
                *ho_numb += 1;

            }
        }
        else
        {
            // vertikal analog
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
