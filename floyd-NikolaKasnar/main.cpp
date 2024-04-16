#include "floyd.h"
#include "clock.h"

#include <iostream>
#include <string>
#include <fstream>
#include <cassert>

// upotreba: main [n] [distance] [side]
// n = broj generiranih gradova 
// distance = udaljenost koja određuje povezanost gradova
// side = duljina stranice kvadrata unutar kojeg se generiraju gradovi
int main(int argc, char * argv[])
{
    // Parametri komandne linije.
    int noPts = 5;
    if(argc > 1)
        noPts = std::stoi(argv[1]);
    double distance = 40;
    if(argc > 2)
        distance = std::stod(argv[2]);
    int side = 100;
    if(argc > 3)
        side = std::stod(argv[3]); // Tu je bila greska, za argument se trebao uzimat argv[3], a ne argv[2]
    assert(distance < side);

    // Generiranje matrice susjedstva.
    Vertices  points;
    generate_vertices(noPts, points, side);
    
    Matrix M;
    generate_edges(distance, points, M);
    std::cout << "Inicijalna matrica:\n";
    // Ispiši inicijalnu matricu.
    std::ofstream out("mat_init.txt");
    if(!out)
        throw std::runtime_error("Ne mogu otvoriti mat_init.txt za pisanje.");
    print(out, M);
    out.close();

    // Napravi kopiju inicijalne matrice jer će ju sekvencijalni algotitam izmijeniti.
    Matrix M1 = M;

    // Sekvencijalna metoda.
    Clock clock;
    minimum_distance(M);
    auto [time, unit] = clock.stop(Clock::microsec);
    std::cout << "Sequential time = " << time << unit << "\n";
    out.open("mat_seq.txt");
    if(!out)
        throw std::runtime_error("Ne mogu otvoriti mat_seq.txt za pisanje.");
    print(out, M);
    out.close();

    // Paralelna metoda.
    clock.start();
    minimum_distance_par(M1);
    auto [time1, unit1] = clock.stop(Clock::microsec);
    std::cout << "Parallel time = " << time1 << unit1 << "\n";
    out.open("mat_par.txt");
    if(!out)
        throw std::runtime_error("Ne mogu otvoriti mat_par.txt za pisanje.");
    print(out, M1);
    out.close();

    return 0;
}
