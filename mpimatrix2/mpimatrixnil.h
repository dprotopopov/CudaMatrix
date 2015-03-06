/*
 * Нулевая матрица
 * Каждый процесс сохраняет свой диапазон ячеек
 * Используется функция fwrite
 */
#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include "mpimatrixfile.h"

#ifndef __MPIMATRIXNIL_H
#define __MPIMATRIXNIL_H

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
void mpi_matrix_nil(
	int height, int width, // Размеры матрицы
	char *outputFileName, 
	long *counter) // Счётчик количества операций
{
	int nrank;     /* Общее количество процессов */
	int myrank;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &nrank);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	if(myrank>0) return;

	assert(myrank==0);

	mpiMatrixHeader header;
	MPI_Datatype dataType = MPI_DOUBLE;

	header.fourCC[0] = 'M';
	header.fourCC[1] = 'P';
	header.fourCC[2] = 'I';
	header.fourCC[3] = 'M';
	header.height = height;
	header.width = width;
	header.dataType = dataType;
	header.offset = sizeof(mpiMatrixHeader);

	FILE *file = fopen(outputFileName,"wb");
	if(file==NULL) { fprintf(stderr,"file open error (%s)\n",outputFileName); fflush(stderr); return; }
	fwrite(&header,1,sizeof(mpiMatrixHeader),file);

	int total = width*height;
	int bufferSize = max(width,height);
	int count = min(width,height);
	T nil = (T)0;
	T *buffer = (T*)malloc(sizeof(T)*bufferSize+1);

	for(int i=0;i<bufferSize;i++) buffer[i]=nil;

	for(int i=0;i<count;i++){
		fwrite(buffer,sizeof(T),bufferSize,file);
	}

	free(buffer);
	fclose(file);

	*counter += total;
}

#endif
