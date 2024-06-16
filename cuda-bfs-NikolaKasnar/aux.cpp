#include "labirint_io.h"
// NE TREBA MIJENJATI

// Provjera korektnosti ulaza
void check_input(LabIOMatrix const & mat, int start_row, int start_col,
	             int stop_row, int stop_col)
{
    std::cout << "Dimenzije matrice : " << mat.rows() <<" x "<< mat.cols() << "\n";
	std::cout << "Broj čvorova je   : " << mat.no_blocks() << ", od " << mat.cols()*mat.rows() << "\n";
    std::cout << "Staza od pozicije ("+ std::to_string(start_row)+","+std::to_string(start_col)+") do pozicije ("
                 +  std::to_string(stop_row)+","+std::to_string(stop_col)+")\n";
    
    if(start_row <0)
        throw std::runtime_error("start_row < 0");
    if(start_col <0)
        throw std::runtime_error("start_col < 0");

    if(stop_row <0)
        throw std::runtime_error("stop_row < 0");
    if(stop_col <0)
        throw std::runtime_error("stop_col < 0");

    if(mat.rows()<= start_row)
        throw std::runtime_error("mat.rows() <= start_row");
    if(mat.cols()<= start_col)
        throw std::runtime_error("mat.cols() <= start_col");

    if(mat.rows()<= stop_row)
        throw std::runtime_error("mat.rows() <= stop_row");
    if(mat.cols()<= stop_col)
        throw std::runtime_error("mat.cols() <= stop_col");

    if(mat(start_row, start_col)  < 0)
        throw std::runtime_error("Pozicija ("+ std::to_string(start_row)+","+std::to_string(start_col)+") je u zidu.");
    if(mat(stop_row, stop_col)  < 0)
        throw std::runtime_error("Pozicija ("+ std::to_string(stop_row)+","+std::to_string(stop_col)+") je u zidu.");

}

