#include <iostream>
#include <vector>
#include <cmath>
#include <mpi.h>

int main(int argc, char** argv) {
    // Inicialização do MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Validação: O número de processos deve ser uma potência de 2 (2^d)
    // Calculamos d (dimensão)
    int d = 0;
    while ((1 << d) < size) d++;

    if ((1 << d) != size) {
        if (rank == 0) {
            std::cerr << "Erro: O número de processos (" << size 
                      << ") deve ser uma potência de 2 para o Hypercube." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Questão 1: Inicialização do token (jeton)
    int token = 0;
    if (rank == 0) {
        token = 42; // Valor arbitrário escolhido
    }

    // Barreira para garantir que todos comecem a medição juntos (Questão 5)
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // ======================================================
    // ALGORITMO DE DIFUSÃO HYPERCUBE (Questões 1, 2, 3 e 4)
    // ======================================================
    // Loop pelas dimensões i de 0 até d-1
    for (int i = 0; i < d; ++i) {
        int mask = 1 << i; // Equivalente a 2^i
        
        // Lógica:
        // Na etapa i, quem tem rank < 2^i já possui o dado e envia para (rank + 2^i).
        // Quem tem rank entre 2^i e 2^(i+1) recebe o dado de (rank - 2^i).
        
        if (rank < mask) {
            // SOU REMETENTE
            int dest = rank + mask;
            // std::cout << "[Passo " << i+1 << "] No " << rank << " enviando para " << dest << std::endl;
            MPI_Send(&token, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
        } 
        else if (rank < (mask * 2)) {
            // SOU DESTINATÁRIO
            int source = rank - mask;
            MPI_Recv(&token, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // std::cout << "[Passo " << i+1 << "] No " << rank << " recebeu " << token << " de " << source << std::endl;
        }
        
        // Sincronização opcional a cada etapa para garantir a ordem "em ondas"
        // (Removemos para medir a performance real na Questão 5, 
        // mas para debug visual pode descomentar)
        // MPI_Barrier(MPI_COMM_WORLD); 
    }

    // Fim da medição de tempo (Questão 5)
    MPI_Barrier(MPI_COMM_WORLD); // Garante que todos terminaram antes de parar o relógio
    double end_time = MPI_Wtime();

    if (rank == 0) {
        std::cout << "--- Difusão Completa em " << d << " etapas ---" << std::endl;
        std::cout << "Token final: " << token << std::endl;
        std::cout << "Tempo decorrido: " << (end_time - start_time) << " segundos." << std::endl;
    }
// Adicione isso antes do return 0
    printf("Rank %d diz: Eu tenho o token %d\n", rank, token);
    MPI_Finalize();
    return 0;
}
