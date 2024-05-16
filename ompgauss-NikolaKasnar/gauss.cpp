#include <iostream>
#include <cmath>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "clock.h"

// Gaussov algoritam za rješavanje sustava.
// A je matrica reda N spremljena kao jednodimenzionalno polje.
// Matrica je zapisana po recima.
// b = vektor dimenzije N. Na ulazu je desna strana, a na izlazu 
// rješenje sustava. 
void gauss(int N, double * A, double *b)
{
    // Petlja u kojoj svodimo matricu na gornjetrokutastu
    // Paraleliziramo vanjsku petlju
    // U slucaju poziva ./main_seq, #pragma pozivi ce se ignorirati te ce se program izvesti sekvencijalno
    for (int k = 0; k < N - 1; ++k) {
        // Sa shared() označavamo koji su resursi dijeljeni izmedu dretvi
        // Sa schedule() oznacim dynamic raspodjelu posla, svaka dretva dobije neki dio posla te kad zavrsi dobije jos
        // Razlog je taj sto je matrica dijagonalna dominantna pa nebi bas bilo fer da neka dretva dobije manje posla
        #pragma omp parallel for shared(A, b) schedule(dynamic)
        for (int i = k + 1; i < N; ++i) {
            double factor = A[i * N + k] / A[k * N + k];
            for (int j = k; j < N; ++j) {
                A[i * N + j] -= factor * A[k * N + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Petlja u kojoj rjesavamo sustav sa gornjetrokutastom matricom
    // Paraleliziramo unutarnju petlju
    for (int i = N - 1; i >= 0; --i) {
        double sum = 0.0;
        // shared() ima istu funkciju kao i u prosloj petlji, a schedule() je ovaj put static
        // Time podijelimo posao ravnomjerno na sve dretve
        #pragma omp parallel for reduction(+:sum) shared(A, b) schedule(static)
        for (int j = i + 1; j < N; ++j) {
            sum += A[i * N + j] * b[j];
        }
        b[i] = (b[i] - sum) / A[i * N + i];
    } 
}


// Kreiraj matricu. Dijagonalno dominantna M-matrica.
// Vraća matricu reda N zapisanu u jedno polje po recima.
double * make_mat(int N)
{
    double c = 0.01;
    double * mat = new double[N*N];
    for(int i=0; i<N*N; ++i) mat[i] = -c;
    for(int i=0; i<N; ++i)
        mat[i*N+i] = N*c +1.0;
    return mat;
}

// Izračunaj desnu stranu iz zadanog točnog rješenja.
// N = red matrice,
// A = matrica zapisana po recima,
// x = točno rješenje.
double * rhs(int N, double const * A, double const * x){
    auto mat = [A, N](int row, int col)-> double { return A[row*N + col]; };
    double * b = new double [N];
    for(int i=0; i<N; ++i){
        b[i] = 0.0;
        for(int j=0; j<N; ++j) 
            b[i] += mat(i,j)*x[j];
    }
    return b;
}


double l2_error(int N, double const * a, double const * b){
    double err = 0.0;
    for(int i=0; i<N; ++i)
        err += (a[i]-b[i])*(a[i]-b[i]);
    return std::sqrt(err);
}

// Glavni program ne treba mijenjati. 
// Argument komandne linije: red sustava. 
int main(int argc, char * argv[])
{
    int N = 5; 
    if(argc > 1)
        N = std::stoi(argv[1]);
    
    // Točno rješenje.
    double * x = new double[N];
    for(int i=0; i<N; ++i)
        x[i] = i+0.5;

    // Generiraj matricu i desnu stranu. 
    double * mat = make_mat(N);
    double *b = rhs(N, mat, x);
    
    Clock clock;
    gauss(N, mat, b);
    auto [elapsed, unit] = clock.stop(Clock::millisec);
    std::cout << "Vrijeme za Gaussove eliminacije =  " << elapsed << unit << "\n";

    std::cout << "l2 greška = " << l2_error(N,x,b) << "\n";

    delete [] mat;
    delete [] x;
    delete [] b;

    return 0;
}
