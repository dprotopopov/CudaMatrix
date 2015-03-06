/*
* Копирование матрицы
* Каждый процесс читает и сохраняет свой диапазон ячеек
* Используется функции fread и fwrite
*/
#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include "mpimatrixfile.h"

#ifndef __MPIMATRIXMOV_H
#define __MPIMATRIXMOV_H

#ifndef MINMAX
#define MINMAX

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

#endif  /* MINMAX */

template <typename T>
void mpi_matrix_mov(
	char *inputFileName, 
	char *outputFileName, 
	long *counter) // Счётчик количества операций
{
	int nrank;     /* Общее количество процессов */
	int myrank;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &nrank);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	if(myrank>0) return;

	FILE *file1 = fopen(inputFileName,"rb");
	FILE *file2 = fopen(outputFileName,"wb");
	if(file1==NULL) { fprintf(stderr,"file open error (%s)\n",inputFileName); fflush(stderr); return; }
	if(file2==NULL) { fprintf(stderr,"file open error (%s)\n",outputFileName); fflush(stderr); return; }
	mpiMatrixHeader header;

	fread(&header,1,sizeof(mpiMatrixHeader),file1);
	fwrite(&header,1,sizeof(mpiMatrixHeader),file2);

	assert(header.offset==sizeof(mpiMatrixHeader));

	int height = header.height;
	int width = header.width;

	int total = width*height;
	int bufferSize = min(width,height);
	int count = max(width,height);

	T *buffer = (T*)malloc(sizeof(T)*bufferSize+1);

	for(int i=0;i<count;i++){
		fread(buffer,sizeof(T),bufferSize,file1);
		fwrite(buffer,sizeof(T),bufferSize,file2);
	}

	free(buffer);

	fclose(file1);
	fclose(file2);

	*counter += total;
}

#endif
