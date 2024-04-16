#include "floyd.h"
#include "random_int.h"

#include <cassert>
#include <cmath>
#include <thread>
#include <unordered_set> // Za izbjegavanje duplikata
#include <iostream> // Za testiranje ispisa vektora
#include <fstream> // Za ispis matrice
#include <algorithm> // Za funkciju min()
#include <limits> // Za maksimalni double u funkciji generate_edges()
#include <vector>


// Cijela implementacija dolazi ovje.
// VAŠ KOD.

// Hash funkcija za std::pair<int, int>, bez toga mi ne radi ideja sa unordered set
struct PairHash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2> &pair) const {
        auto hash1 = std::hash<T1>{}(pair.first);
        auto hash2 = std::hash<T2>{}(pair.second);
        return hash1 ^ hash2;
    }
};

void generate_vertices(std::size_t n, Vertices & points, int side) {
    // Napravimo random number generatore za x i y koordinate
    RandomInt<int> random_x(0, side);
    RandomInt<int> random_y(0, side);

    // Koristimo unordered set za izbjegabanje duplikata
    std::unordered_set<std::pair<int, int>, PairHash> unique_points;

    // Generiramo n vrhova(gradova) i stavimo ih u skup
    while (unique_points.size() < n) {
        int x = random_x();
        int y = random_y();
        unique_points.insert({x, y});
    }

    // Kopiramo jedinstvene vrhove u vektor points
    points.assign(unique_points.begin(), unique_points.end());

    // Testni ispis za vrhove
    /*for (std::size_t i = 0; i < points.size(); ++i)
    {
        std::cout << points[i].first << " , " << points[i].second << std::endl;
    }*/ 
}

void generate_edges(double distance, Vertices const & points, Matrix & M) {
    // Inicijalizaram pocetnu matricu koja sadrzi samo nule
    M.assign(points.size(), std::vector<double>(points.size(), std::numeric_limits<double>::max()));

    // Izracunam udaljenosti i ako je udaljenost manja od "distance" stvorimo rub sa tom vrijednosti(za to koristim std::hypot koja racuna "hipotenuzu")
    // U slucaju da je neka udaljenost veca od "distance" na to mjesto pisemo najvecu vrijednost od double(to je napravljeno u inicijalizaciji matrice sa std::numeric_limits<double>::max()) 
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = 0; j < points.size(); ++j) {
            if (i == j){
                M[i][j] = 0; // Udaljenost vrha od samog sebe je nula
            }
            else{
                double dist = std::hypot(points[i].first - points[j].first, points[i].second - points[j].second);
                if (dist <= distance) {
                    M[i][j] = std::ceil(dist); // Duljinu zakruzimo na najblizi cijeli broj jer tako trazi zadatak
                }
            }
        }
    }
}

// Implementacija funkcije minimum_distance koja sekvencijalno koristi Floyd-Warshallov algoritam
void minimum_distance(Matrix & M) {
    std::size_t n = M.size();

    // Koristimo algoritam kako je opisan u tekstu zadatka
    for(int k=0; k<n; ++k){
        for(int i=0; i<n; ++i){
            for(int j=0; j<n; ++j){
                M[i][j] = std::min(M[i][j], M[i][k]+M[k][j]);
                // std::cout << "M[" << i << "][" << j << "] = " << M[i][j] << std::endl;
            }
        }
    }
}

// Funkcija koja azurira redove u matrici posto tako radi navedeni paralelni algoritam
void update_rows(int start, int end, Matrix &M, int k) {
    for(int i = start; i < end; ++i) {
        for(int j = 0; j < M.size(); ++j) {
            M[i][j] = std::min(M[i][j], M[i][k] + M[k][j]);
        }
    }
}

// Razlika imedu ovog i sekvencijalnog algoritma se pocinje osjecati tek oko 150 vrhova(ako su ostale varijable u komandnoj liniji ostavljene na defaultu)
void minimum_distance_par(Matrix &M) {
    std::size_t n = M.size();

    // Odredimo koliko dretvi koristimo
    int num_threads = std::thread::hardware_concurrency();

    //printf("%d", num_threads);

    // Prolazim kroz sve k vrhove
    std::vector<std::thread> threads;
    for(int k = 0; k < n; ++k) {
        // Izracunamo velicinu dijela matrice koji racuna svaka dretva
        int chunk_size = n / num_threads;
        
        // Pokrecenmo dretve sa funkcijom update_rows
        int start = 0;
        int end = chunk_size;
        for(int i = 0; i < num_threads; ++i) {
            if(i == num_threads - 1) {
                end = n; // Moram postaviti kraj za zadnju dretvu
            }
            threads.emplace_back(update_rows, start, end, std::ref(M), k);
            start = end;
            end += chunk_size;
        }

        // Pricekamo ostale dretve da zavrse
        for(auto &thread : threads) {
            thread.join();
        }
        threads.clear(); // Ocistimo vektor threads za sljedecu iteraciju
    }
}

// Ovdje je alternativna implementacija koju sam pronasao online i malo modificirao, u ovom slucaju ne paraleliziramo drugu petlju
// Takoder izvodi algoritam paralelno i to cak brze od ove implementacije gdje paraleliziramo drugu petlju

// Funkcija worker koja racuna udaljenosti za dio vrhova grafa
// Uzima matricu, pocetak i kraj dijela vrhova za koje racuna udaljenost
/*void floyd_worker(Matrix &M, std::size_t start, std::size_t end) {
    std::size_t n = M.size();
    for (std::size_t k = 0; k < n; ++k) {
        for (std::size_t i = start; i < end; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                M[i][j] = std::min(M[i][j], M[i][k]+M[k][j]);
            }
        }
    }
}

// Svaka dretva obraduje dio vrhova
void minimum_distance_par(Matrix & M) {
    std::size_t n = M.size();
    std::size_t num_threads = std::thread::hardware_concurrency();

    std::vector<std::thread> threads;
    std::size_t chunk_size = n / num_threads;

    // Pokrecem dretve
    for (std::size_t i = 0; i < num_threads; ++i) {
        std::size_t start = i * chunk_size;
        std::size_t end = (i == num_threads - 1) ? n : start + chunk_size;
        threads.emplace_back(floyd_worker, std::ref(M), start, end);
    }

    // Cekamo ostale dretve da zavrse sa poslom
    for (std::thread &t : threads) {
        t.join();
    }
}*/

// Funkcija za printanje matrice
void print(std::ostream & out, Matrix const & M) {
    for (const auto& row : M) {
        for (double value : row) {
            // U slucaju da je vrijednost beskonactnost(tj. std::numeric_limits<double>::max()) ispisemo znak x
            if (value == std::numeric_limits<double>::max()) {
                out << 'x' << " ";
            } else {
                out << value << " ";
            }
        }
        out << "\n";
    }
}