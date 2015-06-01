/*
* Единичная матрица
* Каждый процесс сохраняет свой диапазон ячеек
* Используется функция fwrite
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mpimatrixfile.h"
#include "minmax.h"

#ifndef __MPIMATRIXONE_H
#define __MPIMATRIXONE_H

template <typename T>
void mpi_matrix_one(
	int rank, 
	char *outputFileName, 
	FILE *log, long *counter) // Счётчик количества операций
{
	int np;    /* Общее количество процессов */
	int mp;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &mp);

	if(mp>0) return;

	assert(mp==0);

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

	// Начало записи лога
	fprintf(log,"process %d of %d\n", mp, np);
	fprintf(log,"function %s\n", __FUNCTION__); // http://stackoverflow.com/questions/679021/how-to-find-the-name-of-the-current-function-at-runtime

	for(int j=0;j<rank-1;j++){
		fwrite(buffer,sizeof(T),bufferSize,file);
		fprintf(log,"result:\t"); for(int i=0;i<bufferSize;i++) fprintf(log,"%le\t", (double)buffer[i]); fprintf(log,"\n");
	}


	if(rank>0) {
		fwrite(&one,1,sizeof(T),file);
		fprintf(log,"result:\t"); fprintf(log,"%le\t", (double)one); fprintf(log,"\n");
	}

	fflush(log);
	// Завершение записи лога

	free(buffer);
	fclose(file);

	*counter += total;

}

#endif
