/*
* Копирование матрицы
* Каждый процесс читает и сохраняет свой диапазон ячеек
* Используется функции fread и fwrite
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mpimatrixfile.h"
#include "minmax.h"

#ifndef __MPIMATRIXMOV_H
#define __MPIMATRIXMOV_H

template <typename T>
void mpi_matrix_mov(
	char *inputFileName, 
	char *outputFileName, 
	FILE *log, long *counter) // Счётчик количества операций
{
	int np;    /* Общее количество процессов */
	int mp;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &mp);

	if(mp>0) return;

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

	// Начало записи лога
	fprintf(log,"process %d of %d\n", mp, np);
	fprintf(log,"function %s\n", __FUNCTION__); // http://stackoverflow.com/questions/679021/how-to-find-the-name-of-the-current-function-at-runtime

	for(int j=0;j<count;j++){
		fread(buffer,sizeof(T),bufferSize,file1);
		fprintf(log,"operand:\t"); for(int i=0;i<bufferSize;i++) fprintf(log,"%le\t", (double)buffer[i]); fprintf(log,"\n");
		
		fwrite(buffer,sizeof(T),bufferSize,file2);
		fprintf(log,"result:\t"); for(int i=0;i<bufferSize;i++) fprintf(log,"%le\t", (double)buffer[i]); fprintf(log,"\n");
	}

	fflush(log);
	// Завершение записи лога

	free(buffer);

	fclose(file1);
	fclose(file2);

	*counter += total;
}

#endif
