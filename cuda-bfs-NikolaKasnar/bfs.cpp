#include "bfs.h"
#include "sparse.h"
#include <algorithm> // Treba mi za reverse

void find_path(CSCMat const & incidence, int dst, std::vector<int> const & level, std::vector<int> & path)
{
    // U slucaju da ne postoji put od prvog vrha do cilja, ispisemo prikladnu poruku
    if (level[dst] == -1) {
        std::cerr << "Ne postoji put do cilja!" << std::endl;
        return;
    }

    // Broj redova matrice
    int n = incidence.nrows;
    path.clear();
    // Incijalizacija cilja
    int current = dst;

    while (level[current] != 0) {
        // Dodamo trenutni vrh u put(na pocetku je to pocetni vrh)
        path.push_back(current);
        // Prolazim kroz sve susjede i ako je razina sujeda za jedan manja od razine trenutnog vrha, znaci da se nalazi na putu do starta
        // Postavim taj vrh kao trenutni za sljedecu iteraciju petlje
        for (int i = incidence.colPtrs[current]; i < incidence.colPtrs[current + 1]; ++i) {
            int neighbor = incidence.rowIdx[i];
            if (level[neighbor] == level[current] - 1) {
                current = neighbor;
                break;
            }
        }
    }

    // Za kraj dodamo pocetni vrh na put
    path.push_back(current);
    // Obrnemo put tako da nam ide od starta do cilja
    std::reverse(path.begin(), path.end());
}
