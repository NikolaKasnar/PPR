#pragma once
#include <cstddef>  // size_t
#include <vector>
#include <tuple>
#include <ostream> // Za print() funkciju

using Vertices = std::vector<std::pair<int, int>>;

// Matrix je tip koji predstavlja matricu susjedstva. 
// Vama je ostavljeno na izbor da odaberete tip kojim ćete 
// reprezentirati matricu. Dalje se taj tip pojavljuje kao Matrix
// što ćete morati korigirati ovisno o tipu koji izaberete.  

// Matrica susjedstva, matrica prikazana kao vektor u vektoru
using Matrix = std::vector<std::vector<double>>;

// Slučajnim izborom generiraj n različitih točaka u (0,side)x(0,side).
// Konačan broj može biti manji od n ako se jave duplikati. Drugim 
// riječima možete zanemariti duplikate jer nije vjerojatno da će se
// javiti. Vrhove smjestiti u vektor Vertices. 
void generate_vertices(std::size_t n, Vertices & points, int side=1000);

// Poveži putem sve parove generiranih vrhova čija je udaljenost manja od distance. 
// Na osnovu toga kreiraj matricu susjedstva M. 
void generate_edges(double distance, Vertices const & points, Matrix & M);    

// Izračunaj minimalnu udaljenost između parova vrhova prema 
// Floydovom algoritmu. 
void minimum_distance(Matrix & M);

// Izračunaj minimalnu udaljenost između parova vrhova prema 
// Floydovom algoritmu. Paralelna verzija.
void minimum_distance_par(Matrix & M);

// Ispiši maricu susjedstva na izlazni tok.
void print(std::ostream & out, Matrix const & M);
