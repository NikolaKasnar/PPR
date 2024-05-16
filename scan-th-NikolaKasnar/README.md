**Zadatak**. Potrebno je implementirati paralelnu verziju `scan` algoritma. Detalji algoritma dani su na ovoj 
[stranici](https://web.math.pmf.unizg.hr/nastava/ppr/html/zad-scan-th.html). Potrebno je implementirati **scan algoritam s okrupnjenjem** kako je opisan na [stranici](https://web.math.pmf.unizg.hr/nastava/ppr/html/zad-scan-th.html). Algoritam radi s vektorom cijelih brojeva kao ulaznim nizom i računa niz parcijalnih suma koristeći obično zbrajanje cijelih brojeva kao binarnu operaciju. 

Paralelni scan algoritam radi sa  `nhard = std::thread::hardware_concurrency()` programskih niti ili s dvije niti ako ta funkcija vraća nulu. Cijeli ulazni niz se podijeli na `nhard` dijelova i svaka programska nit radi na jednom dijelu prema opisu algoritma. Za komunikaciju među programskim nitima, koja je neophodna u drugom koraku algoritma, treba koristiti `std::promise<>` i `std::future<>` tipove. Signatura paralelnog algoritma je 


`void parallel_scan(std::vector<int> & data);`

i ona vraća niz parcijalnih suma kroz ulazni vektor. 

Jedan dio testova je već napisan u `main()` funkciji, a tamo je mjesto i za vaše dodatn testove. Kod treba dopuniti samo u datoteci `parallel_scan.cpp`. 
