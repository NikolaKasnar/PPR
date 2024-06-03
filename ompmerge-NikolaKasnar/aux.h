#pragma once

#include <iostream>

// Pomoćne rutine. Tu ne treba ništa mijenjati.

// Provjeri je li niz sortiran u rastućem poretku.
// Niz je  a[lo],a[lo+1],...,a[hi].
template <typename T>
bool is_sorted(const T * a, int lo, int hi){
	for(int i = lo; i<hi; ++i){
           if(a[i] > a[i+1])
			   return false;
	}
    return true;
}

// Ispiši niz na standardni izlaz.
template <typename T>
void print(const T * a, int lo, int hi, const char * text = "a: "){
	std::cout << text;
    if(hi>lo) std::cout << a[lo];
	for(int i = lo+1; i< hi; ++i)
		std::cout<<"," << a[i] ;
	 std::cout << ".\n";
}


