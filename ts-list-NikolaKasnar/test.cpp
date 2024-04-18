#include "ts_list.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <cassert>
#include <numeric>
#include <vector> // Nedostajao je ovaj library za koristenje

// NE DIRATI OVAJ KOD! Ovi testovi moraju ispravno raditi.

using namespace std::literals;

void front(List * plist, int n, int shift=0){
      for(int i=0; i<n; ++i)
          plist->push_front(i+shift);
}

void back(List * plist, int n){
      for(int i=0; i<n; ++i)
          plist->push_back(n+i);
}

void back_remove(List * plist){
    for(int i=0; i<30; ++i){
        int rez = plist->remove(30+i);
        plist->push_back(60+i);
        assert(rez == 1);
    }
}

void remove1(List * plist){
    for(int i=0; i<30; ++i){
        int rez = plist->remove(i);
        assert(rez == 1);
    }
}

void push2(List * lst, int N){
     for(int i=0; i<N; i+=2){
        lst->push_back(i);
        lst->push_front(i+1);
     }
}
void remove2(List * lst, int N, int & cnt){ 
     for(int i=0; i<N; i+=2){
        cnt += lst->remove(i);
        cnt += lst->remove(i+1);
     }
}
 

int main(){

    List lst;

    std::thread t1(front, &lst, 30, 0);
    std::thread t2(back, &lst, 30);
    t1.join();
    t2.join();

    std::cout << "size = " << lst.size() << "\n";
    assert(lst.size() == 60);
    std::cout << "lista nakon front i back:\n";
    lst.print(std::cout);
    std::cout << "\n";
    assert(lst.contains(59));
    assert(lst.contains(0));


    t1 = std::thread(back_remove, &lst);
    t1.join();
    std::cout << "lista nakon back_remove:\n";
    lst.print(std::cout);
    std::cout << "\n";

    t1 = std::thread(remove1, &lst);
    t2 = std::thread(front, &lst, 30, 30);
    t1.join();
    t2.join();

    std::cout << "lista nakon remove i front:\n";
    lst.print(std::cout);
    std::cout << "\n";

    std::cout << "lista nakon remove i find_and_change:\n";
    int cnt0 = lst.find_and_change(89, 100);
    assert(cnt0 == 1);
    lst.print(std::cout);
    std::cout << "\n";


    std::vector<std::thread> thrs(40);
    std::vector<int> cnt(20,0);
    List lst1;
    // 20 threadova, svaki generira 1000 elemenata = 20'000 elemenata
    for(int i=0; i<20; ++i){
        thrs[i]   = std::thread(push2,   &lst1, 1000);
        thrs[39-i] = std::thread(remove2, &lst1, 1000, std::ref(cnt[i]));
    }

    for(int i=0; i<40; ++i)
        thrs[i].join();

    std::cout << "lista nakon 40 niti: push2, remove2:\n";  
    lst1.print(std::cout);
    std::cout << "\n";
    int erased = std::accumulate(cnt.begin(), cnt.end(),0);
    int rest = lst1.size();
    std::cout << "Broj obrisanih elemenata = " << erased 
              << ", preostali = " << rest << "\n";
    assert(rest+erased == 20'000);
    return 0;
}
