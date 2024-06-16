#include "matrix.h"

// Testiraj punu matricu.
int main(){
  Matrix<float> m(3,3);
  m(0,0) = 1;
  m(1,2) = 2;
  m(2,0) = 3;
  print(m);
  assert(m.cols() == 3);
  assert(m.rows() == 3);

  Matrix<int> mi(2,4);
  mi(0,1) = mi(0,2) = 7;
  mi(1,0) = 5; mi(1,3) = -7;
  print(mi);
  assert(mi.cols() == 4);
  assert(mi.rows() == 2);
  return 0;
}
