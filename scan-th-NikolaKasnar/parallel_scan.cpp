#include <vector>
#include <thread>
#include <cassert>
#include <iostream>
#include <future>

void parallel_scan(std::vector<int> & data)
{
    int num_threads = std::thread::hardware_concurrency();
    //std::cout << num_threads << std::endl;
    if (num_threads == 0) // Ako hardware_concurrency() vrati 0, koristi dvije dretve
        num_threads = 2;

    int part_size = data.size() / num_threads;
    //std::cout << part_size << std::endl;

    // Koristim vektor koji sadrzi tip future
    std::vector<std::future<void>> futures(num_threads);

    // Koristim i vektor koji sadrzi tip promise
    std::vector<std::promise<void>> promises(num_threads);

    // Podijelimo vector na jednake dijelove
    for (int i = 0; i < num_threads; ++i) {
        int start = i * part_size;
        int end = (i == num_threads - 1) ? data.size() : (i + 1) * part_size;
        // Ulovimo promise po referenci
        auto& promise = promises[i];
        futures[i] = std::async(std::launch::async, [start, end, &data, &promise, part_size, i] () mutable { // Ulovimo part_size, i
            // Napravimo sekvencijalni scan algoritam na svakom dijelu
            for (int j = start + 1; j < end; ++j) {
                data[j] += data[j - 1];
            }
            promise.set_value();
        });
    }

    // Pricekamo sve dretve da zavrse
    for (auto& future : futures) {
        future.wait();
    }

    // Testni ispis
    /*std::cout << "Data nakon prvog koraka:" << std::endl;
    for (int i = 0; i < data.size(); i++)
    {
        std::cout << data[i] << std::endl;
    }
    std::cout << std::endl;*/

    // Drugi korak algoritma napravimo sekvencijalno
    for (int i = 1; i < num_threads; ++i) {
        data[(i * part_size) + part_size - 1] += data[(i * part_size) - 1];
    }

    // Testni ispis
    /*std::cout << "Data nakon drugog koraka:" << std::endl;
    for (int i = 0; i < data.size(); i++)
    {
        std::cout << data[i] << std::endl;
    }
    std::cout << std::endl;*/

    // Na kraju azuriramo sve ostale elemente niza paralelno
    // Start je prvi element u particiji, a end predzadnji
    for (int i = 1; i < num_threads; ++i) {
        int start = i * part_size;
        int end = ((i + 1) * part_size < data.size()) ? ((i + 1) * part_size - 1) : data.size()-1;
        // Ulovimo promise po referenci
        auto& promise = promises[i];
        // Ulovimo prijasnji promise po referenci
        auto& prevPromise = promises[i - 1];
        futures[i] = std::async(std::launch::async, [start, end, &data, &promise, &prevPromise, part_size, i] () mutable { // Ulovimo part_size, i
            prevPromise.get_future().wait(); // Cekamo prethodnu particiju da zavrsi
            for (int j = start; j < end; ++j) {
                data[j] += data[(i - 1) * part_size + part_size - 1];
            }
            promise.set_value();
        });
    }

    // Opet pricekamo dretve da zavrse
    for (auto& future : futures) {
        future.wait();
    }
}
