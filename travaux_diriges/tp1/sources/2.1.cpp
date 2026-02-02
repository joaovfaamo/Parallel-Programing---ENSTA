#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, nbp;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nbp);

    int token;
    int tag = 0;

    if (rank == 0) {
        token = 1;
        // Envia para o processo 1
        MPI_Send(&token, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
        std::cout << "Processus 0 : envoi initial " << token << " vers 1." << std::endl;

        // Recebe do último processo para fechar o anel
        MPI_Recv(&token, 1, MPI_INT, nbp - 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::cout << "Processus 0 : reçu final " << token << " de " << nbp - 1 << "." << std::endl;

    } else {
        // Recebe do anterior (rank - 1)
        MPI_Recv(&token, 1, MPI_INT, rank - 1, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        token++; // Incrementa

        // Envia para o próximo (usa módulo para voltar ao 0 se for o último)
        int dest = (rank + 1) % nbp;
        MPI_Send(&token, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}