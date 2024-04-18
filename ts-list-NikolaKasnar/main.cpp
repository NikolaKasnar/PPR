#include "ts_list.h"
#include <iostream>
#include <thread>
#include <vector> // Potreban mi je za vector threads
#include <cassert> // Potreban za assert

// Funkcija za obavljanje vise paralelnih operacija
void testFunkcija(List* lst) {
    int rez = 0;
    for (int i = 0; i < 1000; ++i) {
        lst->push_back(i);
        lst->push_front(i);
        lst->contains(i);
        //printf("test");
    }
}

int main(){
// VAŠI TESTOVI

    // Testovi koji ne sadrze threads niti rade ista paralelno, sluze samo za testiranje funkcija
    List lst;

    // Testiram  push_front() i push_back()
    for (int i = 0; i < 5; ++i) {
        lst.push_front(i);
        lst.push_back(i + 5); 
    }

    // Test za remove()
    int removed = lst.remove(3);
    std::cout << "Maknuo " << removed << " elementa vrijednosti 3\n";

    // Test za find_and_change()
    int changed = lst.find_and_change(2, 20);
    std::cout << "Promijenio " << changed << " elementa sa vrijednosti 2 u vrijednost 20\n";

    // Test za contains()
    std::cout << "Lista sadrzi 5: " << std::boolalpha << lst.contains(5) << "\n";
    std::cout << "Lista sadrzi 15: " << std::boolalpha << lst.contains(15) << "\n";

    // Test za size()
    std::cout << "Velicina liste je: " << lst.size() << "\n";

    // Test za print()
    std::cout << "Lista sadrzi: ";
    lst.print(std::cout);
    std::cout << "\n";


    // Testovi za testiranje koda pomocu threads
    List lst2;

    int numThreads = 20;

    // Vector za spremanje threads
    std::vector<std::thread> threads;

    // Pokrecem dretve
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(testFunkcija, &lst2);
    }

    // Pricekam da sve zavrse
    for (auto& thread : threads) {
        thread.join();
    }

    // Ukupni broj operacija koje smo napravili
    //std::cout << "Total operations performed: " << rez << std::endl;

    // Konacna velicina liste
    assert(lst2.size() == 40'000);
    std::cout << "Konacna velicina liste: " << lst2.size() << std::endl;
   
    return 0;
}
