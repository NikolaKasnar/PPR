#pragma once 
// NE TREBA NIŠTA MIJENJATI
#include <vector>
#include <cassert>
#include <iostream>
#include <string>

// Matrica koja sadrži labirint.
// U blokovima u kojima je vrijednost strogo manja od nule (-1) nalazi se prepreka. Ostali 
// blokovi su prazni. Radi se o punoj matrici koja predstavlja pomoćnu strukturu
// za konstrukciju labirinta iz kojeg se zatim konstruira pripadni graf. 
// Blokovi koji nisu prepreke predstavljaju vrhove grafa i stoga ih u konstruktoru 
// numeriramo od 0 do maksimalnog broja vrhova (isključivo). To znači da je prepreka  
// određena brojem manjim od nule, a prazno mjesto brojem >= 0. Klasa LabIOMatrix stoga daje 
// enumeraciju svih čvorova grafa, odnosnodaje  preslikavanja pozicije (i,j) u labirintu 
// u indeks čvora k u grafu. [To bi preslikavanje moglo biti trivijalno ako bismo prepreke
// tretirali kao izolirane čvorove grafa, što ovdje ne radimo.] Prepreke smo izbacili 
// iz grafa. 
class LabIOMatrix{
	public:
	    // Konstrukcija praznog labirinta-
		LabIOMatrix();
        // rezerviraj memoriju za matricu dimenzije nrows x ncols
		void resize(int nrows, int ncols);  
        // Pročitaj labirint iz datoteke
        void read(std::string const & fileName);  

		// Konstruktor kopije. 
        LabIOMatrix(LabIOMatrix const & mat2);
        // Operator pridruživanja.
		LabIOMatrix & operator=(LabIOMatrix && mat2);

        // Operatori dohvata.
		int &       operator()(int i, int j)       { return mdata[i*mncols+j]; }
		int const & operator()(int i, int j) const { return mdata[i*mncols+j]; }

		~LabIOMatrix() { delete [] mdata; }

		int rows() const { return mnrows; }
		int cols() const { return mncols; }

		// broj elemenata u kojima nije prepreka = broj čvorova grafa. 
		int no_blocks() const;

        // Klasa LabIOMatrix enumerira prazne blokove. Metoda find() za zadani indeks 
		// praznog bloka val nalazi indekse retka i stupca u kojima se blok nalazi.
		// To je nužno za iscrtavanje.  Metoda vraća true, osim ako val nije vrh u 
        // grafu, kada vraća false. 
		bool find(int val, int & row, int & col) const;
		// Isprintaj u datoteku koristeći '.' za prazno mjesto 'x' za prepreku.
        // Stazu path ispisuje znakom 'o'.
		void print_ascii(std::string file_name, std::vector<int> path) const;
	private:
	 int mnrows;
	 int mncols;
     int * mdata;
};

std::string base_name(std::string const & name);

// Matrica incidencije grafa koji čini labirint. Ovdje se pamti puna matrica susjedstva.
// Blok unutar labirinta je povezan s lijevim, desnim, gornjim i donjim blokom ukoliko taj 
// blok ne predstavlja prepreku. 
class IncidenceMat{
	public:
		IncidenceMat(LabIOMatrix const &);
		IncidenceMat(int nrows) : mnrows(nrows), mncols(nrows){
			mdata = new int[mnrows*mncols];
			for(int i=0; i<mnrows*mncols; ++i) mdata[i] = 0;
		}
		int & operator()(int i, int j) { return mdata[i*mncols+j]; }
		int const & operator()(int i, int j) const { 
			assert(i >= 0);
			assert(i<mnrows); 
			assert(j>=0);
			assert(j<mncols);
			return mdata[i*mncols+j]; 
			}
		~IncidenceMat() { if(mdata) delete [] mdata; }
		int rows() const { return mnrows; }
		int cols() const { return mncols; }
		void print(std::ostream &) const;
		
	private:
	 int mnrows;
	 int mncols;
     int * mdata;
};

