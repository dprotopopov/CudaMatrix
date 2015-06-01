/*
* Умножение матрицы на число (Multiply To Value)
* Число передаётся как первая матрица размера 1х1
* Каждый процесс читает и сохраняет свой диапазон ячеек
* Используются функции fread и fwrite
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mpimatrixfile.h"
#include "minmax.h"

#ifndef __MPIMATRIXMTV_H
#define __MPIMATRIXMTV_H

template <typename T>
void mpi_matrix_mtv(
	char *inputFileName1, 
	char *inputFileName2, 
	char *outputFileName, 
	FILE *log, long *counter) // Счётчик количества операций
{
	int np;    /* Общее количество процессов */
	int mp;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &mp);

	FILE *file=NULL;
	mpiMatrixHeader header;
	T value;

	if(mp==0) file = fopen(inputFileName1,"rb");
	if(mp==0) fread(&header,1,sizeof(mpiMatrixHeader),file);
	if(np>1) MPI_Bcast(&header,sizeof(mpiMatrixHeader),MPI_BYTE,0,MPI_COMM_WORLD);

	assert(header.height==1);
	assert(header.width==1);
	assert(header.offset==sizeof(mpiMatrixHeader));

	MPI_Datatype dataType = header.dataType;
	MPI_Offset offset = header.offset;

	if(mp==0) fread(&value,sizeof(T),1,file);
	if(np>1) MPI_Bcast(&value,1,dataType,0,MPI_COMM_WORLD);
	if(mp==0) fclose(file);

	if(mp==0) file = fopen(inputFileName2,"rb");
	if(mp==0) fread(&header,1,sizeof(mpiMatrixHeader),file);
	if(np>1) MPI_Bcast(&header,sizeof(mpiMatrixHeader),MPI_BYTE,0,MPI_COMM_WORLD);

	assert(header.dataType==dataType);
	assert(header.offset==sizeof(mpiMatrixHeader));

	int width = header.width;
	int height = header.height;

	int total = width*height;
	int wrank = min(np,total); // Количество реально используемых процессов

	// Создаём группу из реально используемых процессов и коммуникатор этой группы
	MPI_Group world_group;
	MPI_Group group;
	MPI_Comm comm;
	int *ranks = (int *)malloc(sizeof(int)*wrank);
	for(int i=0;i<wrank;i++) ranks[i]=i;
	MPI_Comm_group(MPI_COMM_WORLD,&world_group);
	MPI_Group_incl(world_group,wrank,ranks,&group);
	MPI_Comm_create(MPI_COMM_WORLD,group,&comm);
	free(ranks);

	if(mp>=wrank) return;

	int start = total*mp/wrank;
	int end = total*(mp+1)/wrank;
	int length=end-start;
	int bufferSize = (total+wrank-1)/wrank;
	int bufferSize1 = length;

	T *buffer = (T*)malloc(sizeof(T)*bufferSize+1);
	T *buffer1 = (T*)malloc(sizeof(T)*bufferSize1+1);

	// Чтение файла
	for(int j=0;j<wrank;j++) {
		int start = total*j/wrank;
		int end = total*(j+1)/wrank;
		int count=end-start;
		if(mp==0) fread(buffer,sizeof(T),count,file);
		if(j==0 && mp==0) memcpy(buffer1,buffer,length*sizeof(T));
		else if(mp==0) MPI_Send(buffer,count,dataType,j,0,comm);
		else if(mp==j) MPI_Recv(buffer1,length,dataType,0,0,comm,MPI_STATUS_IGNORE);
	}

	if(mp==0) fclose(file);

	// Начало записи лога
	fprintf(log,"process %d of %d\n", mp, np);
	fprintf(log,"function %s\n", __FUNCTION__); // http://stackoverflow.com/questions/679021/how-to-find-the-name-of-the-current-function-at-runtime

	fprintf(log,"operand 1:\n"); fprintf(log,"%le\n", (double)value);
	fprintf(log,"operand 2:\t"); for(int i=0;i<length;i++) fprintf(log,"%le\t", (double)buffer1[i]); fprintf(log,"\n");

	// Операция умножения на число
	for(int i=0;i<length;i++) buffer1[i]*=value;

	fprintf(log,"result:\t"); for(int i=0;i<length;i++) fprintf(log,"%le\t", (double)buffer1[i]); fprintf(log,"\n");

	fflush(log);
	// Завершение записи лога

	if(mp==0) file = fopen(outputFileName,"wb");
	if(mp==0) fwrite(&header,1,sizeof(mpiMatrixHeader),file);

	// Запись файла результата
	for(int j=0;j<wrank;j++) {
		int start = total*j/wrank;
		int end = total*(j+1)/wrank;
		int count=end-start;
		if(j==0 && mp==0) memcpy(buffer,buffer1,length*sizeof(T));
		else if(mp==0) MPI_Recv(buffer,count,dataType,j,0,comm,MPI_STATUS_IGNORE);
		else if(mp==j) MPI_Send(buffer1,length,dataType,0,0,comm);
		if(mp==0) fwrite(buffer,sizeof(T),count,file);
	}

	if(mp==0) fclose(file);

	free(buffer);
	free(buffer1);

	*counter += length;
}

#endif
