#include <iostream>
#include <string>
#include "bfs.h"
#include "sparse.h"
#include "labirint_io.h"
#include <cuda_runtime.h>
#include <vector>
#include <cuda.h> // Zaboravljeno staviti

void check_input(LabIOMatrix const & mat, int start_row, int start_col,
	             int stop_row, int stop_col);


// Ovu implementaciju sam koristio prije jer sam prvotno mislio da ne trebamo koristiti jezgru sa predavanja
/*__global__ void bfsKernel(int *d_rowPtrs, int *d_colIdx, int *d_level, int *d_frontier, int frontierSize, int level) {
    // id dretve
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < frontierSize) {
        // Trenutni vrh grafa koji obradujemo
        int node = d_frontier[idx];
        // Prolazim kroz sve susjede
        for (int edge = d_rowPtrs[node]; edge < d_rowPtrs[node + 1]; ++edge) {
            int neighbor = d_colIdx[edge];
            if (d_level[neighbor] == -1) { // Ako susjed nije posjecen dodamo +1
                d_level[neighbor] = level + 1;
            }
        }
    }
}

void bfs(CSRMat const &csr_incidence, int startIdx, std::vector<int> &level) {
    // Broj redova u csr matrici
    int n = csr_incidence.nrows;
    // Razina pretrazivanja, na neposjecene vrhove stavimo -1, to su svi na pocetku osim pocetnog koji je 0
    level.resize(n, -1);
    level[startIdx] = 0;

    // Inicijaliziram sve potrebno na GPU
    int *d_rowPtrs, *d_colIdx, *d_level, *d_frontier;
    cudaMalloc(&d_rowPtrs, (n + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, csr_incidence.nelem * sizeof(int));
    cudaMalloc(&d_level, n * sizeof(int));

    // Kopiram matricu i level na GPU
    cudaMemcpy(d_rowPtrs, csr_incidence.rowPtrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, csr_incidence.colIdx, csr_incidence.nelem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, level.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // U frontieru cuvamo sve posjecene vrhove, na pocetku se u njemu nalazi samo pocetni vrh
    std::vector<int> frontier = { startIdx };
    cudaMalloc(&d_frontier, n * sizeof(int));
    cudaMemcpy(d_frontier, frontier.data(), frontier.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Pocetni level je 0
    int level_num = 0;
    while (!frontier.empty()) {
        // Racunamo potrebni broj blockova te pokrenemo jezgru sa njima
        int frontierSize = frontier.size();
        int blocks = (frontierSize + 255) / 256;
        bfsKernel<<<blocks, 256>>>(d_rowPtrs, d_colIdx, d_level, d_frontier, frontierSize, level_num);
        
        cudaDeviceSynchronize();

        // Azuriramo frontier sa svim novim vrhovima koje smo posjetili
        std::vector<int> new_frontier;
        cudaMemcpy(level.data(), d_level, n * sizeof(int), cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) {
            if (level[i] == level_num + 1) {
                // Dodamo vrhove u vector
                new_frontier.push_back(i);
            }
        }

        // U sljedecoj iteraciji saljemo azurirani vector
        frontier = new_frontier;
        // Kopiramo ga na frontier na GPU
        cudaMemcpy(d_frontier, frontier.data(), frontier.size() * sizeof(int), cudaMemcpyHostToDevice);
        ++level_num;
    }

    // Pocistimo memoriju
    cudaFree(d_rowPtrs);
    cudaFree(d_colIdx);
    cudaFree(d_level);
    cudaFree(d_frontier);
}*/

// Implementacija jezgre slicna onoj iz materijala za predavanja samo drugacije prenosim matricu
__global__ void bfs_kernel(int *d_rowPtrs, int *d_colIdx, int *d_level, int *d_prev_front, int *d_curr_front, const int *d_prev_front_size, int *d_curr_front_size, int currentLevel) {
    // id dretve
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Selektiraju se samo programske niti koje odgovaraju elementima iz polja prev_front
    if (i < *d_prev_front_size) {
        // Uzima se indeks vrha s prethodne fronte
        int vertex = d_prev_front[i];
        // Iteriramo kroz sve susjede tog vrha
        for (int edge = d_rowPtrs[vertex]; edge < d_rowPtrs[vertex + 1]; ++edge) {
            int neighbor = d_colIdx[edge];
            if (atomicCAS(&d_level[neighbor], -1, currentLevel + 1) == -1) {  // Usporedimo i zamijenimo
                // Ako susjed nije posjecen dodamo ga u frontier
                int idx = atomicAdd(d_curr_front_size, 1);
                d_curr_front[idx] = neighbor;
            }
        }
    }
}

// Funkcija za racunanje BFS-a
void bfs(CSRMat const &csr_incidence, int startIdx, std::vector<int> &level) {
    // Broj redova u csr matrici
    int n = csr_incidence.nrows;
    // Razina pretrazivanja, na neposjecene vrhove stavimo -1, to su svi na pocetku osim pocetnog koji je 0
    level.resize(n, -1);
    level[startIdx] = 0;

    // Inicijaliziram sve potrebno na GPU
    int *d_rowPtrs, *d_colIdx, *d_level, *d_prev_front, *d_curr_front, *d_prev_front_size, *d_curr_front_size;
    cudaMalloc(&d_rowPtrs, (n + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, csr_incidence.nelem * sizeof(int));
    cudaMalloc(&d_level, n * sizeof(int));
    cudaMalloc(&d_prev_front, n * sizeof(int));
    cudaMalloc(&d_curr_front, n * sizeof(int));
    cudaMalloc(&d_prev_front_size, sizeof(int));
    cudaMalloc(&d_curr_front_size, sizeof(int));

    // Kopiram matricu i level na GPU
    cudaMemcpy(d_rowPtrs, csr_incidence.rowPtrs, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, csr_incidence.colIdx, csr_incidence.nelem * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_level, level.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    // U frontieru cuvamo sve posjecene vrhove, na pocetku se u njemu nalazi samo pocetni vrh
    std::vector<int> prev_front = { startIdx };
    int prev_front_size = 1;
    cudaMemcpy(d_prev_front, prev_front.data(), prev_front_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_prev_front_size, &prev_front_size, sizeof(int), cudaMemcpyHostToDevice);

    // Pocetni level je 0
    int level_num = 0;
    while (prev_front_size > 0) {
        int curr_front_size = 0;
        cudaMemcpy(d_curr_front_size, &curr_front_size, sizeof(int), cudaMemcpyHostToDevice);

        // Racunamo potrebni broj blokova te pokrenemo jezgru sa njima
        int blocks = (prev_front_size + 255) / 256;
        bfs_kernel<<<blocks, 256>>>(d_rowPtrs, d_colIdx, d_level, d_prev_front, d_curr_front, d_prev_front_size, d_curr_front_size, level_num);

        cudaDeviceSynchronize();

        // Kopiramo novu velicinu frontiera natrag na hosta
        cudaMemcpy(&curr_front_size, d_curr_front_size, sizeof(int), cudaMemcpyDeviceToHost);

        // Kriterij zaustavljanja petlje
        if (curr_front_size == 0) {
            break; // Nema vise vrhova za istrazit
        }

        // U sljedecoj iteraciji saljemo azurirani niz
        prev_front_size = curr_front_size;
        // Kopiramo ih na frontier na GPU
        cudaMemcpy(d_prev_front, d_curr_front, curr_front_size * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_prev_front_size, &curr_front_size, sizeof(int), cudaMemcpyHostToDevice);

        // Povecamo level
        ++level_num;
    }

    // kopiram polje level natrag na host
    cudaMemcpy(level.data(), d_level, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Pocistimo memoriju
    cudaFree(d_rowPtrs);
    cudaFree(d_colIdx);
    cudaFree(d_level);
    cudaFree(d_prev_front);
    cudaFree(d_curr_front);
    cudaFree(d_prev_front_size);
    cudaFree(d_curr_front_size);
}




////////////////////////////////////////////////////////////

int main(int argc, char * argv[])
{
	int start_row = -1; // polazna točka row
	int start_col = -1; // polazna točka col
	int stop_row = -1;  // završna točka row
	int stop_col = -1;  // završna točka col
	std::string file_name = "labirint.txt"; // ulazna datoteka s labirintom

	if(argc >= 6){
		start_row = std::stoi(argv[1]);
		start_col = std::stoi(argv[2]);
		stop_row = std::stoi(argv[3]);
		stop_col = std::stoi(argv[4]);
		file_name = argv[5];
	}
	else{
		std::cerr << "Upotreba: " << argv[0] << " start_row start_col stop_row stop_col file_name\n";
		std::cerr << "Brojevi stupaca i redaka idu od nule.\n";
		std::exit(1);
	}

	// Kreiraj labirint. Labirint je zadan s matricom tipa LabMatrix.
	LabIOMatrix mat;
    mat.read(file_name);
    check_input(mat, start_row, start_col, stop_row, stop_col);

	// Kreiraj graf iz labirinta. Funkcija vraća matricu incidencije koja je ovdje dana kao 
	// puna matrica. 
	IncidenceMat incidence(mat);  
	CSRMat csr_incidence(incidence);
	CSCMat csc_incidence(incidence);

//	  csr_incidence.print();
//    csc_incidence.print();

	int start_idx = mat(start_row, start_col);
	int stop_idx  = mat(stop_row,stop_col);
	std::cout << "start index = " << start_idx << ", stop index = " << stop_idx << "\n";

    /// VAŠ CUDA kod  DOLAZI OVDJE /////////////////////////////////////////
    // ALOCIRAJ MEMORIJU NA GPU, KOPIRAJ PODATKE S CPU NA GPU, 
    // POZOVI JEZGRU, KOPIRAJ LEVEL POLJE S GPU NA CPU.

    // Pozivamo funkciju za BFS
    std::vector<int> level;
    bfs(csr_incidence, start_idx, level);

    ///////////////////////////////////////////////////////////////////////

    std::vector<int>  path;  // STAZA
    // IZRAČUNAJ STAZU
	find_path(csc_incidence, stop_idx, level, path); 
    // PRINTAJ STAZU U DATOTEKU
    mat.print_ascii("out_"+base_name(file_name), path);

   // POČISTITE MEMORIJU //////////////////// 
    // Memoriju pocistim u funkciji bfs
    ///////////////////////////////////////////
    return 0;
}