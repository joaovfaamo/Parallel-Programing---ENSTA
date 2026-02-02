#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath> // Necessário para std::abs e cálculos matemáticos
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Número total de pontos (aumente para mais precisão)
    long long total_points = 100000000; 
    long long local_points = total_points / size;
    long long local_count = 0;

    // Semente única por processo
    unsigned int seed = time(NULL) + rank * 100;

    double start_time = MPI_Wtime();

    // Loop de Monte Carlo Local
    for (long long i = 0; i < local_points; ++i) {
        double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;

        if (x * x + y * y <= 1.0) {
            local_count++;
        }
    }

    // Soma os resultados parciais
    long long global_count = 0;
    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        double pi = 4.0 * (double)global_count / (double)total_points;
        std::cout << "--- Resultado MPI ---" << std::endl;
        std::cout << "Pi estimado: " << pi << std::endl;
        std::cout << "Erro: " << std::abs(pi - 3.1415926535) << std::endl;
        std::cout << "Tempo: " << end_time - start_time << " segundos." << std::endl;
    }

    MPI_Finalize();
    return 0;
}