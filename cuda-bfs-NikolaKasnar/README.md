 

**Zadatak**. Potrebno je napraviti program koji nalazi najkraći put između dvije točke 
unutar labirinta. Labirint je matrica od NxM blokova među kojima neki predstavljaju prepreke. 
Moguće je kretati se samo horizontalno i vertikalno (ali ne i dijagonalno). 

Labirinti koje treba koristiti u testiranju dani su u dvije datoteke, `labirint1.txt` i `labirint2.txt`. U prvoj je
veći labirint (90x100 blokova), a u drugoj manji (10x10 blokova). Prazno mjesto u labirintu je označeno s '.', a prepreka s 'x'.

Polazeći od labirinta treba konstruirati pripadni graf. Svaki prazni blok u labirintu je vrh u grafu koji je povezan s onim od svoja četiri susjeda (lijevi, desni, gornji i donji) koji nisu prepreke. Pripadnu matricu susjedstva treba zapisati i u CSR i u CSC formatu.

Treba odabrati polazni (slobodni) blok u labirintu, odnosno jedan vrh grafa, i napraviti pretraživanje u širinu na grafičkoj kartici. Rezultat algoritma je polje razina svih vrhova koje treba prebaciti na CPU. Pri tome koristiti jezgru koja optimizira broj pokrenutih programskih niti 
pomoću frontova. 

Odabrati završni vrh grafa i koristeći CSC zapis matrice susjedstva naći najkraću stazu od polaznog do krajnjeg vrha. Taj dio treba obaviti na CPU. Rezultat treba zapisati u datoteku `labirint1A.txt` odnosno `labirint2A.txt` tako što će 
staza biti označena znakom 'o' (slobodni blok s '.', a prepreka s 'x'). 

Program konstruirati tako da uzima 5 parametara: `red1  stupac1 red2 stupac2 ime_ulazne_datoteke`.
Tu je `(red1  stupac1)` polazni blok, a `(red2 stupac2)` završni blok. Labirint će biti pročitan iz ulazne datoteke.  

**Detalji**: Čitanje labirinta u ulaznu matricu `LabIOMatrix` i zatim u punu matricu susjedstva `IncidenceMat` vam je zadano. Datoteke `labirint_io.h`, `labirint_io.cpp` ne treba mijenjati. Jednako tako, zadan  vam je kod za konstrukciju CSR matrice iz pune matrice. Za CSC matricu taj kod trebate dopuniti sami. Trebate napisati rutinu za nalaženje staze, BSF jezgru i dopuniti `main` funkciju.
