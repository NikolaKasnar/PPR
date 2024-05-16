#include <vector>
#include <iostream>

#include "random_int.h"
#include "clock.h"
#include "aux.h"

// Paralelna verzija scan operacije daje rezultat kroz ulazno polje.
// data = na ulazu polje podataka
// data = na izlazu polje parcijalnih suma ulaznih podataka.
void parallel_scan(std::vector<int> & data);


int main()
{
    ///////////
    // MJESTO ZA VAŠE TESTOVE
    /*std::vector<int> data_test {1, 3, 2, 0, 1, 3, 4, 1, 2, 5, 2, 3};
    for (int i = 0; i < data_test.size(); i++)
    {
        std::cout << data_test[i] << std::endl;
    }
    std::cout << std::endl << "Sekvencijalni scan:" << std::endl;
    std::vector<int> scan_rand2;
    seq_scan(data_test, scan_rand2);
    for (int i = 0; i < data_test.size(); i++)
    {
        std::cout << scan_rand2[i] << std::endl;
    }
    
    std::vector<int> par_scan_rand2=data_test;
    parallel_scan(par_scan_rand2);
    std::cout << std::endl << "Paralelni scan:" << std::endl;
    for (int i = 0; i < data_test.size(); i++)
    {
        std::cout << par_scan_rand2[i] << std::endl;
    }


    std::cout << std::endl;*/
    ////////////
    /// TESTOVI KOJI TREBAJU KOREKTNO RADITI
    std::vector<int> data_rand(100000);
    RandomInt rnd(0,10);
    for(int & x : data_rand)
        x = rnd();
    std::vector<int> scan_rand;

    Clock clock;
    seq_scan(data_rand, scan_rand);
    auto [time1, unit1] = clock.stop(Clock::microsec);
    std::cout << "Sekvencijalni scan na " << data_rand.size() << " elemenata traje "
              << time1 << unit1 << "\n";

    std::vector<int> par_scan_rand=data_rand;

    clock.start();
    parallel_scan(par_scan_rand);
    auto [time2, unit2] = clock.stop(Clock::microsec);
    std::cout << "Paralelni scan na " << data_rand.size() << " elemenata traje "
              << time2 << unit2 << "\n";
  
    if(!compare(scan_rand, par_scan_rand))
        std::cerr << "Parallel scan gives WRONG result!\n";
    else
        std::cout << "Parallel scan gives GOOD result!\n";

    if(data_rand.size() < 30){
            print(scan_rand);
            print(par_scan_rand);
    }
      
    return 0;
}
