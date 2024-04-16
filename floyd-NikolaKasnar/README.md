*Zadatak*. Zadatak je zadan na stranici 
[FloydAlgo](https://web.math.pmf.unizg.hr/nastava/ppr/html/Cpp/floydalgo.html).

Paralelizacija Floyd_warshallovog algoritma

Algoritam se sastoji od jedne trostruke petlje. Vanjsku petlju po k
ne možemo paralelizirati jer je proces sekvencijalan. Za izračun vrijednosti dk+1(i,j) moramo prvo izračunati vrijednosti dk(i,j)

.

Tvrdimo da drugu petlju možemo paralelizirati. To znači da za svaki pojedini k
izračun nove matrice ne ovisi o poretku računanja redaka. Retke matrice možemo računati u bilo kojem poretku i stoga ih možemo izračunavati paralelno. To ponovo slijedi iz činjenice da se u koraku k ne mijenjaju vrijednosti S[k][j] (k+1-vi redak) niti S[i][k] (k+1-vi stupac). Svi drugi elementi S[i][j] matrice S

se izračunavaju pomoću elementa S[i][j], S[k][j] i S[i][k]. Odatle slijedi da poredak izračunavanja nije bitan. Time smo pokazali da možemo paralelizirati drugu petlju. Štoviše, umjesto druge petlje možemo paralelizirati treću petlju (što bi dalo slabije rezultate).

Zadatak. Implementirati i testirati Floydov algoritam.

    Prvi se dio zadatka sastoji u generiranju grafa. U kvadratu (0,L)×(0,L)

generiramo n točaka (gradova) slučajnim izborom. Radi jednostavnosti za koordinate gradova koristimo cjelobrojne vrijednosti i uzimamo cjelobrojni L

. Taj dio koda treba implementirati u funkciji generate_vertices() (vidjeti datoteku floyd.h).

Bridove grafa dobivamo na sljedeći način: odaberemo duljinu D
(D<L) i povežemo svaka dva grada koji su na udaljenosti manjoj od D. Na taj način dobivamo bridove grafa i njihove težine (tj. duljine). S tom informacijom treba generirati matricu susjedstva S

    . Sve duljine bridova zaokružite na najbliži cijeli broj radi jednostavnosti usporedbe kasnije. Taj dio programa radi metoda generate_edges().

    Formirati metodu minimum_distance() koja implementira sekvencijalnu verziju Floydovog algoritma.

    Formirati metodu minimum_distance_par() koja implementira paralelnu verziju Floydovog algoritma. Koristite std::thread::hardware_concurrency() broj niti.

    Konstrirajte metodu print() koja ispisuje matricu susjedstva na izlazni tok. Beskonačnu vrijednost možete ispisati kao znak x.

    Testirajte paralelni i serijski algoritam. Kod za testiranje je zadan u datoteci main.cpp. Ono što ćete morati mijenjati u toj datoteci je tip koji koristite za reprezentaciju matrice. U programu se pretpostavlja da je matrica predstavljena klasom Matrix i na taj način se koristi. Vi možete matricu predstaviti na bilo koji način koji vam odgovara, ali tada morate izmijeniti one linije koda koje referiraju matricu.

    Za test korektnosti uzmite usporedbu ispisa matrice dobivene serijskim i paralelnim kodom. Kako je serijski algoritam sasvim jednostavan pretpostavka je da je to dovoljno za testiranje korektnosti paralelnog koda. Matrica susjedstva sadrži samo cijele brojeve (i beskonačno) što pojednostavljuje usporedbu. Na komandnoj liniji usporedbu možete napraviti pomoću naredbe diff. Naredba

dif mat_par.txt mat_seq.txt

treba dati prazan ispis.

