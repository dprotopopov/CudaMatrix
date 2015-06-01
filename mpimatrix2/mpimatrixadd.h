/*
* Сложение матриц
* Каждый процесс считывает свой диапазон ячеек
* Производит сложение ячеек
* Сохраняет вычисленные ячейки
* Используются функции fread и fwrite
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "mpimatrixfile.h"

#ifndef __MPIMATRIXADD_H
#define __MPIMATRIXADD_H

template <typename T>
void mpi_matrix_add(
	char *inputFileName1,
	char *inputFileName2,
	char *outputFileName, 
	FILE *log, long *counter) // Счётчик количества операций
{
	int np;    /* Общее количество процессов */
	int mp;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &mp);

	FILE *file1=NULL;
	FILE *file2=NULL;

	mpiMatrixHeader header1;
	mpiMatrixHeader header2;

	if(mp==0) file1 = fopen(inputFileName1,"rb");
	if(mp==0) fread(&header1,1,sizeof(mpiMatrixHeader),file1);
	if(np>1) MPI_Bcast(&header1,sizeof(mpiMatrixHeader),MPI_BYTE,0,MPI_COMM_WORLD);

	assert(header1.offset==sizeof(mpiMatrixHeader));

	int height = header1.height;
	int width = header1.width;
	int total = width*height;

	MPI_Datatype dataType = header1.dataType;
	MPI_Offset offset = header1.offset;

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

	if(mp==0) file2 = fopen(inputFileName2,"rb");
	if(mp==0) fread(&header2,1,sizeof(mpiMatrixHeader),file2);
	if(np>1) MPI_Bcast(&header2,sizeof(mpiMatrixHeader),MPI_BYTE,0,comm);

	assert(header1.dataType==header2.dataType);
	assert(header1.width==header2.width);
	assert(header1.height==header2.height);
	assert(header1.offset==header2.offset);

	int start = total*mp/wrank;
	int end = total*(mp+1)/wrank;
	int length=end-start;
	int bufferSize = (total+wrank-1)/wrank;
	int bufferSize1 = length;
	int bufferSize2 = length;

	T *buffer = (T*)malloc(sizeof(T)*bufferSize+1);
	T *buffer1 = (T*)malloc(sizeof(T)*bufferSize1+1);
	T *buffer2 = (T*)malloc(sizeof(T)*bufferSize2+1);

	// Чтение первого файла
	for(int j=0;j<wrank;j++) {
		int start = total*j/wrank;
		int end = total*(j+1)/wrank;
		int count=end-start;
		if(mp==0) fread(buffer,sizeof(T),count,file1);
		if(j==0 && mp==0) memcpy(buffer1,buffer,length*sizeof(T));
		else if(mp==0) MPI_Send(buffer,count,dataType,j,0,comm);
		else if(mp==j) MPI_Recv(buffer1,length,dataType,0,0,comm,MPI_STATUS_IGNORE);
	}

	// Чтение второго файла
	for(int j=0;j<wrank;j++) {
		int start = total*j/wrank;
		int end = total*(j+1)/wrank;
		int count=end-start;
		if(mp==0) fread(buffer,sizeof(T),count,file2);
		if(j==0 && mp==0) memcpy(buffer2,buffer,length*sizeof(T));
		else if(mp==0) MPI_Send(buffer,count,dataType,j,0,comm);
		else if(mp==j) MPI_Recv(buffer2,length,dataType,0,0,comm,MPI_STATUS_IGNORE);
	}

	if(mp==0) fclose(file1);
	if(mp==0) fclose(file2);

	// Начало записи лога
	fprintf(log,"process %d of %d\n", mp, np);
	fprintf(log,"function %s\n", __FUNCTION__); // http://stackoverflow.com/questions/679021/how-to-find-the-name-of-the-current-function-at-runtime

	fprintf(log,"operand 1:\t"); for(int i=0;i<length;i++) fprintf(log,"%le\t", (double)buffer1[i]); fprintf(log,"\n");
	fprintf(log,"operand 2:\t"); for(int i=0;i<length;i++) fprintf(log,"%le\t", (double)buffer2[i]); fprintf(log,"\n");
	
	// Операция сложения
	for(int i=0;i<length;i++) buffer1[i]+=buffer2[i];

	fprintf(log,"result:\t"); for(int i=0;i<length;i++) fprintf(log,"%le\t", (double)buffer1[i]); fprintf(log,"\n");

	fflush(log);
	// Завершение записи лога

	if(mp==0) file1 = fopen(outputFileName,"wb");
	if(mp==0) fwrite(&header1,1,sizeof(mpiMatrixHeader),file1);

	// Запись файла результата
	for(int j=0;j<wrank;j++) {
		int start = total*j/wrank;
		int end = total*(j+1)/wrank;
		int count=end-start;
		if(j==0 && mp==0) memcpy(buffer,buffer1,length*sizeof(T));
		else if(mp==0) MPI_Recv(buffer,count,dataType,j,0,comm,MPI_STATUS_IGNORE);
		else if(mp==j) MPI_Send(buffer1,length,dataType,0,0,comm);
		if(mp==0) fwrite(buffer,sizeof(T),count,file1);
	}

	free(buffer);
	free(buffer1);
	free(buffer2);

	if(mp==0) fclose(file1);

	*counter += length;
}

#endif
