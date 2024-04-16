#include <pngwriter.h> 
#include <iostream>
#include <thread>
#include <cmath>

// Funkcija koja racuna Mandebrotov skup za dane brojeve
int mandelbrot(double cx, double cy, int maxIterations) {
    double x = 0.0, y = 0.0;
    int iteration = 0;
    
    // Provjeravamo uvijet ograničenosti i jel broj iteracija i dalje manji od zadanog
    while (x*x + y*y <= 4 && iteration < maxIterations) {
        double xtemp = x*x - y*y + cx;
        y = 2*x*y + cy; // Rekurzija za Mandelbrotove brojeve
        x = xtemp;
        iteration++;
    }
    
    return iteration;
}

// Funkcija za crtanje Mandelbrotov skupa za dobiveno podrucje i spremanje dobivenog PNG dokumenta
void drawMandelbrot(double xmin, double xmax, double ymin, double ymax, const std::string& filename) {
    const int duljina_slike = 800;
    const int visina_slike = 800;
    const int maxIterations = 1000;

    // Konstruktor pisaca
    pngwriter png(duljina_slike, visina_slike, 0, filename.c_str());

    double dx = (xmax - xmin) / duljina_slike;
    double dy = (ymax - ymin) / visina_slike;

    for (int px = 0; px < duljina_slike; ++px) {
        for (int py = 0; py < visina_slike; ++py) {
            double x0 = xmin + px * dx;
            double y0 = ymin + py * dy;
            int color = mandelbrot(x0, y0, maxIterations);
            double grayscale = (double)color / maxIterations;
            png.plot(px, py, grayscale, grayscale, grayscale);
        }
    }

    png.close();
}

int main() {

    // Vrtim beskonacnu petlju sve dok korisnik ne unese "exit"
    while (true) {
        double xmin, xmax, ymin, ymax;
        std::string filename;
        
        // Unos podataka sa konzole
        // za unos rubova skupa brojeve odvojite razmakom
        // Npr. za skup [−2,0.47]×[−1.12,1.12] unos izgleda ovako "-2 0.47 -1.12 1.12"
        // Za ime dateteke mozete unesti npr. "test.png"
        std::cout << "Unesite xmin, xmax, ymin, ymax: ";
        std::cin >> xmin >> xmax >> ymin >> ymax;
        std::cout << "Unesite ime dokumenta u koji zelite spremiti PNG sliku: ";
        std::cin >> filename;

        // Pokrecem dretvu te joj saljem brojeve intervala te ime dokumenta gdje spremamo PNG
        std::thread t(drawMandelbrot, xmin, xmax, ymin, ymax, filename);
        
        t.join();

        // Provjerimo jel korisnik zeli zavrsiti ili nastaviti
        // "exit" za izlazak a bilo koja tipka(osim enter) za nastavak crtanja
        std::cout << "Unesite 'exit' za izaci ili bilo koju drugu tipku za nastaviti: ";
        std::string input;
        std::cin >> input;
        if (input == "exit")
            break;
    }

    return 0;
}
