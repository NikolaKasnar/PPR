#include "labirint_io.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <algorithm>
// NE TREBA NIŠTA MIJENJATI

LabIOMatrix::LabIOMatrix() : mdata(nullptr), mnrows(0), mncols(0) {}

void LabIOMatrix::resize(int nrows, int ncols){
    mnrows = nrows;
    mncols = ncols;
    if(mdata) 
        delete [] mdata;
	mdata = new int[mnrows*mncols];
	for(int i=0; i<mnrows*mncols; ++i)
        mdata[i] = 0;
}

LabIOMatrix::LabIOMatrix(LabIOMatrix const & mat2) : mnrows(mat2.mnrows), mncols(mat2.mncols){
	mdata = new int[mnrows*mncols];
	for(int i=0; i<mnrows*mncols; ++i) mdata[i] = mat2.mdata[i];
}

LabIOMatrix & LabIOMatrix::operator=(LabIOMatrix && mat2){
    delete [] mdata;
	mdata = mat2.mdata;
	mnrows = mat2.mnrows;
	mncols = mat2.mncols;
	mat2.mdata = nullptr;
	mat2.mnrows = 0;
	mat2.mncols = 0;
	return *this;
}

int LabIOMatrix::no_blocks() const{
	int cnt = mnrows*mncols;
    for(int i=0; i<mnrows; ++i)
		for(int j=0; j<mncols; ++j)
			if((*this)(i,j) < 0) 
				--cnt;
	return cnt;
}

bool LabIOMatrix::find(int val, int & row, int & col) const {
      auto it = std::find(mdata, mdata+mnrows*mncols, val); 
	  if(it == mdata+mnrows*mncols)
	      return false;
	  int k = it-mdata;  // udaljenost 
	  row = k / mncols;
	  col = k % mncols;
      return true;
}

void LabIOMatrix::print_ascii(std::string file_name, std::vector<int> path) const {
    std::ofstream file;
	file.open(file_name);
    if(!file)
        throw std::runtime_error("ne mogu otvoriti datoteku " + file_name + " za pisanje.");
	for(int row = 0; row < mnrows; ++row){
		for(int col = 0; col < mncols; ++col){
			int val = (*this)(row,col);
			if(val < 0)
			    file << "x";
			else{
                auto it = std::find(path.begin(), path.end(), val);
                if(it == path.end())
			        file << ".";
                else 
                    file << "o";
            }
		}
		file << "\n";
	}
}

IncidenceMat::IncidenceMat(LabIOMatrix const & labirint) : mnrows(labirint.no_blocks()), mncols(mnrows)
{
    mdata = new int[mnrows*mncols];

	for(int i=0; i<labirint.rows(); ++i)
    {
		for(int j=0; j<labirint.cols(); ++j){
			int index = labirint(i,j);
			if(index < 0)
				continue; // zid
			if(j>0){
			     int indexS = labirint(i,j-1);
				 if(indexS >= 0) 
					 operator()(index,indexS) = 1;
			}
			if(j+1 < labirint.cols()){
			     int indexS = labirint(i,j+1);
				 if(indexS >= 0) 
					 operator()(index,indexS) = 1;
			}
			if(i>0){
			     int indexS = labirint(i-1,j);
				 if(indexS >= 0) 
					 operator()(index,indexS) = 1;
			}
			if(i+1 < labirint.rows()){
			     int indexS = labirint(i+1,j);
				 if(indexS >= 0) 
					 operator()(index,indexS) = 1;
			}
		}
	}
}

void IncidenceMat::print(std::ostream & out) const {
     for(int i=0; i<rows(); ++i){
		 for(int j=0; j<cols(); ++j){
			 out << (*this)(i,j);
		 }
		 out << "\n";
	 }
	 out << "\n";
}

void LabIOMatrix::read(std::string const & fileName)
{
	// Prvo pročitaj broj redaka i stupaca
    std::ifstream in(fileName);
    if(!in)
        throw std::runtime_error("Ne mogu otvoriti " + fileName + " za čitanje.");
    
    int nrows = -1, ncols = -1;
	std::string line;
	int no_lines = 0;
	int no_cols = -1;
	while(std::getline(in, line, '\n')){
        ++no_lines;
		int row_size = line.size();
		if(no_cols == -1) 
			no_cols = row_size;
		if(row_size != no_cols){
			std::cerr << "Red br " << no_lines << " nije jednake duljine kao prethodni." << std::endl;
			std::exit(-1);
		}
	}
    in.close();

    resize(no_lines, no_cols);
    int N = no_lines * no_cols;

    // Pročitaj podatke linearno.
	in.open(fileName);
    if(!in)
        throw std::runtime_error("Ne mogu otvoriti " + fileName + " za čitanje.");

    int k=0;
    for (int i = 0; i < N; ++i){
        char c;
        in >> c;
        if(c == 'x') 
            mdata[i] = -1;
        else
            mdata[i] = k++;
    }

    in.close();
}

std::string base_name(std::string const & name){
    auto n = name.find_last_of("/");
    if(n != std::string::npos)
        return name.substr(n+1);
    return name;
}

