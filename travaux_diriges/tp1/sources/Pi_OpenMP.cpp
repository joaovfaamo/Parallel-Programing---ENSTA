#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>

int main() {
    long long total_points = 100000000; // 100 milhões
    long long global_count = 0;

    double start_time = omp_get_wtime();

    // Região Paralela
    // reduction(+:global_count) cria cópias locais da variável e soma no final
    #pragma omp parallel reduction(+:global_count)
    {
        // Cada thread precisa de sua própria semente
        unsigned int seed = time(NULL) + omp_get_thread_num();

        #pragma omp for
        for (long long i = 0; i < total_points; ++i) {
            double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
            double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;

            if (x * x + y * y <= 1.0) {
                global_count++;
            }
        }
    }

    double end_time = omp_get_wtime();
    double pi = 4.0 * (double)global_count / (double)total_points;

    std::cout << "--- Resultado OpenMP ---" << std::endl;
    std::cout << "Pi estimado: " << pi << std::endl;
    std::cout << "Tempo: " << end_time - start_time << " s" << std::endl;

    return 0;
}