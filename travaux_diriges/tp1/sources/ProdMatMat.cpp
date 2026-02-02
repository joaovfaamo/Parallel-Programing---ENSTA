#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"

namespace {

// TAMANHO DO BLOCO
// Dica: Para o item do exercício "faire varier la taille", 
// mude este valor (ex: 32, 64, 128, 256, 512) e recompile para achar o melhor.
// Geralmente, valores entre 32 e 128 funcionam bem para Cache L1.
const int szBlock = 64; 

// KERNEL: Multiplica um sub-bloco
// Aqui dentro mantemos a ordem KIJ que você descobriu ser a melhor sequencialmente.
// Como os blocos são pequenos (cabem no cache), essa ordem é extremamente rápida.
void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, 
                   const Matrix& A, const Matrix& B, Matrix& C) {
    
    // Limites de segurança para não estourar a matriz nas bordas
    int iEnd = std::min(A.nbRows, iRowBlkA + szBlock);
    int jEnd = std::min(B.nbCols, iColBlkB + szBlock);
    int kEnd = std::min(A.nbCols, iColBlkA + szBlock);

    // Ordem KIJ (K externo, J interno)
    for (int k = iColBlkA; k < kEnd; ++k) {
        for (int i = iRowBlkA; i < iEnd; ++i) {
            // Otimização extra opcional: Salvar A(i,k) em variável local
            // double aik = A(i, k); 
            for (int j = iColBlkB; j < jEnd; ++j) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

} // namespace


Matrix operator*(const Matrix& A, const Matrix& B) {
    Matrix C(A.nbRows, B.nbCols, 0.0);
    
    // ESTRATÉGIA DE BLOCOS + OPENMP
    // Esses laços externos "navegam" pela matriz pulando de bloco em bloco.
    
    // 1. Paralelizamos o 'ib' (linhas de C). 
    // Como cada thread cuida de linhas diferentes de C, não há Race Condition.
    #pragma omp parallel for
    for (int ib = 0; ib < A.nbRows; ib += szBlock) {
        
        // 2. Laço jb (colunas de C)
        for (int jb = 0; jb < B.nbCols; jb += szBlock) {
            
            // 3. Laço kb (dimensão comum K) - O Somatório dos blocos
            for (int kb = 0; kb < A.nbCols; kb += szBlock) {
                
                // Chamamos a função rápida para calcular apenas este pedacinho
                prodSubBlocks(ib, jb, kb, A, B, C);
            }
        }
    }
    return C;
}