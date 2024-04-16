**Zadatak**. Treba riješiti zadatak zadan na stranici 
[Mandelbrot](https://web.math.pmf.unizg.hr/nastava/ppr/html/Cpp/zad-mandelbrot.html).

Zadatak

U programima koji imaju grafičko sučelje iscrtavanje sučelja i distribucija događaja se tipično vrše u glavnoj programskoj niti. Sve duže operacije koje bi mogle usporiti ili blokirati grafičko sučelje vrše se stoga u sporednim programskim nitima. U ovom zadatku ćemo simulirati jednu takvu situaciju. Kako ne bismo morali koristiti grafičku biblioteku kao što je Qt ili wxWidgets, naš glavni program će u beskonačnoj petlji čitati korisničke podatke s konzole. U posebnoj programskoj niti ćemo iscrtavati Mandelbrotov skup u PNG datoteku koristeći PNGWriter biblioteku (vidjeti niže).

Glavni program učitava s komandne linije pravokutnik [xmin,xmax]×[ymin,ymax]

koji predstavlja dio kompleksne ravnine koju iscrtavamo. Pored toga učitava ime datoteke u koju želimo spremiti PNG sliku. Nakon toga u zasebnoj programskoj niti pokreće program za iscrtavanje PNG slike dijela Mandelbrotovog skupa u odabranom području. Sve se to dešava u beskonačnoj petlji i korisnik mora imati mogućnost izaći iz petlje, što prekida program.

Cijeli program treba biti u datoteci mandelbrot.cpp.

