**Zadatak**. Koristeći OpenMP kreirati paralelnu verziju Gaussovih eliminacija za rješavanje linearnog sustava. Pretpostavka je da je matrica kvadratna i da pivotiranje nije potrebno. Mala rekapitulacija Gausovih eliminacija je dana [ovdje](https://web.math.pmf.unizg.hr/nastava/ppr/html/gaussove_eliminacije.html).

Zadatak je dopuniti kod u repozitoriju i napisati komentare i rezultate u datoteku **REZULTATI.txt**. Serijski i paralelni kod moraju biti u jednoj datoteci i razlika mora biti samo u opciji prevoditelju.

Gausove se eliminacije sastoje od dvije faze: u prvoj se matrica sustava svodi na gornju trokutastu matricu, a drugoj se rješava sustav s gornjom trokutastom matricom.  Potrebno je paralelizirati **obje** faze.

Svođenje na gornji trokut (prva faza) se sastoji od trostruke petlje. Koju je od tri petlje moguće paralelizirati, a koju nije moguće? Rješavanje trokutastog sustava se sastoji od dvije petlje. Koju je petlju moguće paralelizirati, a koju ne? Ovdje trebate biti svjesni da se rješavanje trokutastog sustava može organizirati na dva načina koji se razlikuju u poretku petlji. Odaberite način koji je najpovoljniji za paralelizaciju. 

Ispravnost koda se testira tako što se zadaje rješenje i pomoću njega se generira desna strana. Nakon rješavanja sustava se ispisuje greška.

Gausove eliminacije sadrže u sebi brojne _zavisnosti_. Stoga je potrebno naći **tip raspodjele posla** koji daje najbolje rezultate (odredba `schedule`). 

Potrebno je mjeriti ubrzanje paralelnog koda u odnosu na serijski. Izmjerite ubrzanje koje postižete na 2 i 4 procesora na matricama reda 200, 500, 1000, 1500 i 2000. 

U datoteku **REZULTATI.txt** zapisati:

 - odabrani najefikasniji tip raspodjele posla;
 - dobivena ubrzanja;
 - koje su petlje odabrane za paralelizaciju i zašto;
 - koji je efekt paralelizacije dijela algoritma koji riješava sustav.  