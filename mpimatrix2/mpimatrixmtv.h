/*
* Умножение матрицы на число (Multiply To Value)
* Число передаётся как первая матрица размера 1х1
* Каждый процесс читает и сохраняет свой диапазон ячеек
* Используются функции fread и fwrite
*/
#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include "mpimatrixfile.h"

#ifndef __MPIMATRIXMTV_H
#define __MPIMATRIXMTV_H

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
void mpi_matrix_mtv(
	char *inputFileName1, 
	char *inputFileName2, 
	char *outputFileName, 
	long *counter) // Счётчик количества операций
{
	int nrank;     /* Общее количество процессов */
	int myrank;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &nrank);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	FILE *file=NULL;
	mpiMatrixHeader header;
	T value;

	if(myrank==0) file = fopen(inputFileName1,"rb");
	if(myrank==0) fread(&header,1,sizeof(mpiMatrixHeader),file);
	if(nrank>1) MPI_Bcast(&header,sizeof(mpiMatrixHeader),MPI_BYTE,0,MPI_COMM_WORLD);

	assert(header.height==1);
	assert(header.width==1);
	assert(header.offset==sizeof(mpiMatrixHeader));

	MPI_Datatype dataType = header.dataType;
	MPI_Offset offset = header.offset;

	if(myrank==0) fread(&value,sizeof(T),1,file);
	if(nrank>1) MPI_Bcast(&value,1,dataType,0,MPI_COMM_WORLD);
	if(myrank==0) fclose(file);

	if(myrank==0) file = fopen(inputFileName2,"rb");
	if(myrank==0) fread(&header,1,sizeof(mpiMatrixHeader),file);
	if(nrank>1) MPI_Bcast(&header,sizeof(mpiMatrixHeader),MPI_BYTE,0,MPI_COMM_WORLD);

	assert(header.dataType==dataType);
	assert(header.offset==sizeof(mpiMatrixHeader));

	int width = header.width;
	int height = header.height;

	int total = width*height;
	int wrank = min(nrank,total); // Количество реально используемых процессов

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

	if(myrank>=wrank) return;

	int start = total*myrank/wrank;
	int end = total*(myrank+1)/wrank;
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
		if(myrank==0) fread(buffer,sizeof(T),count,file);
		if(j==0 && myrank==0) memcpy(buffer1,buffer,length*sizeof(T));
		else if(myrank==0) MPI_Send(buffer,count,dataType,j,0,comm);
		else if(myrank==j) MPI_Recv(buffer1,length,dataType,0,0,comm,MPI_STATUS_IGNORE);
	}

	if(myrank==0) fclose(file);

	// Операция умножения на число
	for(int i=0;i<length;i++) buffer1[i]*=value;

	if(myrank==0) file = fopen(outputFileName,"wb");
	if(myrank==0) fwrite(&header,1,sizeof(mpiMatrixHeader),file);

	// Запись файла результата
	for(int j=0;j<wrank;j++) {
		int start = total*j/wrank;
		int end = total*(j+1)/wrank;
		int count=end-start;
		if(j==0 && myrank==0) memcpy(buffer,buffer1,length*sizeof(T));
		else if(myrank==0) MPI_Recv(buffer,count,dataType,j,0,comm,MPI_STATUS_IGNORE);
		else if(myrank==j) MPI_Send(buffer1,length,dataType,0,0,comm);
		if(myrank==0) fwrite(buffer,sizeof(T),count,file);
	}

	if(myrank==0) fclose(file);

	free(buffer);
	free(buffer1);

	*counter += length;
}

#endif
