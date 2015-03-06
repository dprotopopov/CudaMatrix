/*
* Транспонирование матрицы
* Каждый процесс считывает свой диапазон колонок
* Транспонирует матрицу из этих колонок
* Сохраняет транспонированную матрицу из этих колонок
* Используются функции fread fwrite
*/
#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include "mpimatrixfile.h"

#ifndef __MPIMATRIXROT_H
#define __MPIMATRIXROT_H

template <typename T>
void mpi_matrix_rot(
	char *inputFileName, 
	char *outputFileName, 
	long *counter) // Счётчик количества операций
{
	int nrank;     /* Общее количество процессов */
	int myrank;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &nrank);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	FILE *file=NULL;
	mpiMatrixHeader header;

	if(myrank==0) file = fopen(inputFileName,"rb");
	if(myrank==0) fread(&header,1,sizeof(mpiMatrixHeader),file);
	if(nrank>1) MPI_Bcast(&header,sizeof(mpiMatrixHeader),MPI_BYTE,0,MPI_COMM_WORLD);

	assert(header.offset==sizeof(mpiMatrixHeader));

	int height1 = header.height;
	int width1 = header.width;

	MPI_Datatype dataType = header.dataType;
	MPI_Offset offset = header.offset;

	int wrank = min(nrank, width1); // Количество реально используемых процессов

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

	int height2 = width1;
	int width2 = height1;

	int start = width1*myrank/wrank;
	int end = width1*(myrank+1)/wrank;
	int length=end-start;
	int bufferSize=max(width1,width2);
	int bufferSize1=length*height1;
	int bufferSize2=width2*length;

	T *buffer = (T*)malloc(sizeof(T)*bufferSize+1);
	T *buffer1 = (T*)malloc(sizeof(T)*bufferSize1+1);
	T *buffer2 = (T*)malloc(sizeof(T)*bufferSize2+1);

	// Считываем построчно исходный файл
	// Каждый процесс считывает только свой диапазон колонок
	for(int i=0;i<height1;i++){
		if(myrank==0) fread(buffer,sizeof(T),width1,file);
		for(int j=0;j<wrank;j++) {
			int start = width1*j/wrank;
			int end = width1*(j+1)/wrank;
			int count=end-start;
			if(j==0 && myrank==0) memcpy(&buffer1[i*length],buffer,length*sizeof(T));
			else if(myrank==0) MPI_Send(&buffer[start],count,dataType,j,0,comm);
			else if(myrank==j) MPI_Recv(&buffer1[i*length],length,dataType,0,0,comm,MPI_STATUS_IGNORE);
		}
	}

	if(myrank==0) fclose(file);

	// Транспонируем колонки в строки
	for(int i=0;i<height1;i++){
		for(int j=0;j<length;j++){
			buffer2[j*width2+i]=buffer1[i*length+j];
		}
	}

	header.height=height2;
	header.width=width2;

	if(myrank==0) file = fopen(outputFileName,"wb");
	if(myrank==0) fwrite(&header,1,sizeof(mpiMatrixHeader),file);

	// Сохраняем построчно результирующий файл
	// Каждый процесс сохраняет только свой диапазон колонок
	for(int j=0;j<wrank;j++) {
		int start = height2*j/wrank;
		int end = height2*(j+1)/wrank;
		int count=end-start;
		for(int i=0;i<count;i++){
			if(j==0 && myrank==0) memcpy(buffer,&buffer2[i*width2],width2*sizeof(T));
			else if(myrank==j) MPI_Send(&buffer2[i*width2],width2,dataType,0,0,comm);
			else if(myrank==0) MPI_Recv(buffer,width2,dataType,j,0,comm,MPI_STATUS_IGNORE);
			if(myrank==0) fwrite(buffer,sizeof(T),width2,file);
		}
	}

	if(myrank==0) fclose(file);

	free(buffer);
	free(buffer1);
	free(buffer2);

	*counter += length*height1;
}

#endif
