#pragma once
#include "sparse.h"
#include <vector>
// NETREBA MIJENJATI

// Nađi (jednu od) najkraću stazu od vrha dst do vrha čiji je level jednak nuli.
//
// csc_incidence = matrica incidencije u CSC formatu
// dst = vrh koji je kraj staze
// level = polje indeksa udaljenosti koje izračuna "breadth first search" algoritam
// path  = staza od dst do src (koji je implicitno zadan s level)
void find_path(CSCMat const & csc_incidence, int dst, 
               std::vector<int> const & level, std::vector<int> & path);

