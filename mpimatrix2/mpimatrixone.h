/*
* Единичная матрица
* Каждый процесс сохраняет свой диапазон ячеек
* Используется функция fwrite
*/
#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include "mpimatrixfile.h"

#ifndef __MPIMATRIXONE_H
#define __MPIMATRIXONE_H

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
void mpi_matrix_one(
	int rank, 
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
	header.height = rank;
	header.width = rank;
	header.dataType = dataType;
	header.offset = sizeof(mpiMatrixHeader);

	FILE *file = fopen(outputFileName,"wb");
	if(file==NULL) { fprintf(stderr,"file open error (%s)\n",outputFileName); fflush(stderr); return; }
	fwrite(&header,1,sizeof(mpiMatrixHeader),file);

	int total = rank*rank;
	int bufferSize = rank+1;
	T nil = (T)0;
	T one = (T)1;
	T *buffer = (T*)malloc(sizeof(T)*bufferSize+1);
	
	buffer[0]=one;
	for(int i=1;i<=rank;i++) buffer[i]=nil;

	for(int i=0;i<rank-1;i++){
		fwrite(buffer,sizeof(T),bufferSize,file);
	}
	if(rank>0) fwrite(&one,1,sizeof(T),file);

	free(buffer);
	fclose(file);

	*counter += total;

}

#endif
