// Beispiel für Aufruf von PIKAIA mit C++-Hauptprogramm
// und C++-Routinen für zu optimierende Funktionen

// Übersetzung der Programmteile:

// f77 -c fortran_part.f oder
// gfortran -c fortran_part.f  (je nach FORTRAN-Compiler)
// g++ fortran_part.o cpp_part.cpp -lm -lf2c -o programm  bzw.
// g++ fortran_part.o cpp_part.cpp -lm -lgfortran -o programm
// Definition der FORTRAN-Parameter: siehe PIKAIA-Manual

#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>

using namespace std;

extern "C++" // Definition der FORTRAN-Routinen pikaia und rninit
{
    void pikaia_(float(int*, float*), int&, float*, float*, float*, int*);
    void rninit_(int&);
};

float func_p1(int *n, float *x){  
    float r = sqrt((x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5));
    const float PI = acos(-1);     
    return cos(9*PI*r)*cos(9*PI*r)*exp(-r*r/0.15);
}

int main(){  // aufrufendes Hauptprogramm
// Initialisierung
    float f; // Fitness
    srand(time(NULL));
    int status; // status
    int n = 2;   // Zahl der Parameter
    int rn=rand();
    float ctrl[]={-1,-1,-1,-1,-1,-1,-1,-1,-1,1,1,0};  // Steuerungsfeld
    rninit_(rn);
    float x[n]; // 2 Parameter
// Aufruf von PIKAIA
    pikaia_(func_p1,n,ctrl,x,&f,&status);
// Auswerten müsst ihr dann selber ...
    printf("x=%f, y=%f\n",x[0],x[1]);
    printf("func_val=%f\n",f);
    printf("status=%i\n",status);
    return 0;
}