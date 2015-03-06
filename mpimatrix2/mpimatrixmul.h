/*
* Умножение матрицы
* Каждый процесс считывает свой диапазон колонок второй матрицы
* Читаем построчно первую матрицу и умножаем на колонки второй матрицы
* Сохраняем полученый вектор
* Используются функции fread fwrite
*/
#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include "mpimatrixfile.h"

#ifndef __MPIMATRIXMUL_H
#define __MPIMATRIXMUL_H

template <typename T>
void mpi_matrix_mul(
	char *inputFileName1, 
	char *inputFileName2, 
	char *outputFileName, 
	long *counter) // Счётчик количества операций
{
	int nrank;     /* Общее количество процессов */
	int myrank;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &nrank);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	FILE *file1=NULL;
	FILE *file2=NULL;

	mpiMatrixHeader header1;
	mpiMatrixHeader header2;

	if(myrank==0) file2 = fopen(inputFileName2,"rb");
	if(myrank==0) fread(&header2,1,sizeof(mpiMatrixHeader),file2);
	if(nrank>1) MPI_Bcast(&header2,sizeof(mpiMatrixHeader),MPI_BYTE,0,MPI_COMM_WORLD);

	int height2 = header2.height;
	int width2 = header2.width;

	MPI_Datatype dataType = header2.dataType;
	MPI_Offset offset = header2.offset;

	int wrank = min(nrank, width2); // Количество реально используемых процессов

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

	if(myrank==0) file1 = fopen(inputFileName1,"rb");
	if(myrank==0) fread(&header1,1,sizeof(mpiMatrixHeader),file1);
	if(nrank>1) MPI_Bcast(&header1,sizeof(mpiMatrixHeader),MPI_BYTE,0,comm);

	assert(header1.dataType==header2.dataType);
	assert(header1.width==header2.height);
	assert(header1.offset==header2.offset);
	assert(header1.offset==sizeof(mpiMatrixHeader));
	assert(header2.offset==sizeof(mpiMatrixHeader));

	int height1 = header1.height;
	int width1 = header1.width;

	int start = width2*myrank/wrank;
	int end = width2*(myrank+1)/wrank;
	int length=end-start;
	int bufferSize=max(width1,width2);
	int bufferSize1=width1;
	int bufferSize2=length*height2;
	int bufferSize3=length;

	T *buffer = (T*)malloc(sizeof(T)*bufferSize+1);
	T *buffer1 = (T*)malloc(sizeof(T)*bufferSize1+1);
	T *buffer2 = (T*)malloc(sizeof(T)*bufferSize2+1);
	T *buffer3 = (T*)malloc(sizeof(T)*bufferSize3+1);

	// Считываем построчно исходный файл
	// Каждый процесс считывает только свой диапазон колонок
	for(int i=0;i<height2;i++){
		if(myrank==0) fread(buffer,sizeof(T),width2,file2);
		for(int j=0;j<wrank;j++) {
			int start = width2*j/wrank;
			int end = width2*(j+1)/wrank;
			int count=end-start;
			if(j==0 && myrank==0) memcpy(&buffer2[i*length],buffer,length*sizeof(T));
			else if(myrank==0) MPI_Send(&buffer[start],count,dataType,j,0,comm);
			else if(myrank==j) MPI_Recv(&buffer2[i*length],length,dataType,0,0,comm,MPI_STATUS_IGNORE);
		}
	}

	if(myrank==0) fclose(file2);

	header2.height = height1;

	if(myrank==0) file2 = fopen(outputFileName,"wb");
	if(myrank==0) fwrite(&header2,1,sizeof(mpiMatrixHeader),file2);

	// Считываем построчно исходный файл
	// Все процессы считывают одинаковые данные
	// Умножаем строку на столбцы в памяти
	// Сохраняем построчно
	// Каждый процесс сохраняет только свой диапазон колонок
	for(int i=0;i<height1;i++){
		if(myrank==0) fread(buffer1,sizeof(T),width1,file1);
		if(nrank>1) MPI_Bcast(buffer1, width1, dataType, 0, comm);

		for(int j=0;j<length;j++){
			T s = (T)0;
			for(int k=0;k<width1;k++){
				s+=buffer1[k]*buffer2[k*length+j];
			}
			buffer3[j]=s;
		}
		
		for(int j=0;j<wrank;j++) {
			int start = width2*j/wrank;
			int end = width2*(j+1)/wrank;
			int count=end-start;
			if(j==0 && myrank==0) memcpy(buffer,buffer3,length*sizeof(T));
			else if(myrank==0) MPI_Recv(&buffer[start],count,dataType,j,0,comm,MPI_STATUS_IGNORE);
			else if(myrank==j) MPI_Send(buffer3,length,dataType,0,0,comm);
		}

		if(myrank==0) fwrite(buffer,sizeof(T),width2,file2);
	}

	if(myrank==0) fclose(file1);
	if(myrank==0) fclose(file2);

	free(buffer);
	free(buffer1);
	free(buffer2);
	free(buffer3);

	*counter += height1*width1*length;
}

#endif
