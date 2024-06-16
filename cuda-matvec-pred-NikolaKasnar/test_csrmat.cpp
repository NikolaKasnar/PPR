#include "csr_mat.h"
#include "matrix.h"
#include "lapmat.h"

int main()
{
   Matrix<float> m(3,3);
   m(0,0) = 1; m(0,1) = 4;
   m(1,2) = 2; m(1,1) = 9;
   m(2,0) = 3; m(2,1) = 7;
   std::cout << "Puna matrica:\n";
   print(m);

   CSRMatrix<float> mcrs;
   convertToCSR(m,mcrs);
   std::cout << "CSR  matrica:\n";
   print(mcrs);
   std::cout << "CSR  kao puna matrica:\n";
   printFull(mcrs);

   unsigned int N = 2;
   std::cout << "Marica Laplaceovog operatora: N = " << N << "\n";
   CSRMatrix<float> lap;
   LaplaceMatrix(N,lap);
   print(lap);

   std::cout << "Puna marica Laplaceovog operatora: N = " << N << "\n";
   printFull(lap);
   return 0;
}
