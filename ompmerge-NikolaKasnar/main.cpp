#include <iostream>
#include <random>
#include <cassert>
#include <omp.h>
#include "aux.h"

// Merge funkcija za spajanje dva niza u jedan
template <typename T>
void merge(T* a, int lo, int mid, int hi) {
    int n1 = mid - lo + 1;
    int n2 = hi - mid;
    T* left = new T[n1];
    T* right = new T[n2];

    for (int i = 0; i < n1; ++i)
        left[i] = a[lo + i];
    for (int i = 0; i < n2; ++i)
        right[i] = a[mid + 1 + i];

    int i = 0, j = 0, k = lo;
    while (i < n1 && j < n2) {
        if (left[i] <= right[j]) {
            a[k++] = left[i++];
        } else {
            a[k++] = right[j++];
        }
    }

    while (i < n1) {
        a[k++] = left[i++];
    }

    while (j < n2) {
        a[k++] = right[j++];
    }

    delete[] left;
    delete[] right;
}

template <typename T>
void mergesort( T * a, int lo, int hi) // granice su uključive
{
	// Nizovi duljine manje od treshold ce se sortat sekvencijalno
    const int threshold = 1000;
    if (lo < hi) {
        int mid = lo + (hi - lo) / 2;
		// U slucaju da je duljina niza manja od tresholda, pozove mergesort sekvencijalno
        // Ovaj dio sam maknuo pri uvodu final odredbe koja mi sluzi kao zamjena za if
        /*if (hi - lo < threshold) {
            mergesort(a, lo, mid);
            mergesort(a, mid + 1, hi);
        }*/
			// Varijable su private za ovaj task, a=pointer na niz, lo=donji indeks za sortiranje, mid=srednji indeks za sortiranje
			// Taj task block onda izvrsava mergesort za te elemente
            // Ako izraz (mid - lo < treshold)) daje vrijednost true tada izvršni sustav više neće generirati nove zadatke
            #pragma omp task firstprivate(a, lo, mid) final(mid - lo < threshold) mergeable
            {
                mergesort(a, lo, mid);
            }
			// Isto kao za prijasnji task samo sada za gornju polovicu
            // Ako zakomentiramo ovu liniju, kao i u materijalima na sluzbenoj stranici, mozemo dobiti malo ubrzanje(to nisam tu napravio)
            // Ideja je da nije potrebno generirati svaki puta dva zadatka, vec jedan od njih možemo izvrsiti u trenutnoj programskoj niti
            #pragma omp task firstprivate(a, mid, hi) final(hi - mid - 1 < threshold) mergeable
            {
                mergesort(a, mid + 1, hi);
            }
			// Pricekamo da obje grane zavrse sa poslom
            #pragma omp taskwait
        //}
		// Na kraju spojimo dva dobivena niza
        merge(a, lo, mid, hi);
    }
}

// Kostur test-programa vam je dan ovdje.
int main(int argc, char * argv[])
{
    if(argc < 2){
		std::cout << "Usage: " << argv[0] << " N [output]\n";
		std::cout << "          N = broj elemenata u nizu.\n";
		std::cout << "          output (proizvoljan) se zadaje kada se želi ispis niza.\n";
		return 1;
	}
    // Učitaj broj elemenata.
	int N = std::atoi(argv[1]);
    assert(N>0);
	std::cout << "# elements = " << N << "\n";
    // Ispis ili ne?
	bool do_output = false;
	if(argc > 2)
		do_output = true;

    // Generator slučajnih brojeva.
    std::random_device rd;
    std::default_random_engine r_engine; 
    r_engine.seed( rd() ); 
	std::uniform_int_distribution<> dist(10,100);

    // Alociraj memoriju i generiraj slučajni niz.
	int * a = new int[N];
    for(int i=0; i<N; ++i)
		a[i] = dist(r_engine);

	if(do_output) print(a, 0, N, "Slučajni niz:\n"); 

	double begin = omp_get_wtime();

    // Ovdje dolazi poziv mergesort algoritmu.
	#pragma omp parallel
    {
        #pragma omp single
        {
            mergesort(a, 0, N - 1);
        }
    }

	double end = omp_get_wtime();
	std::cout << "Proteklo vrijeme = " << end-begin << " sec.\n"; 
	std::cout << "Niz je sortiran : " << std::boolalpha <<  is_sorted(a, 0, N-1) << "\n";
    
	if(do_output) print(a, 0, N, "Poslije sortiranja:\n");
    
	delete [] a;
	return 0;
}
