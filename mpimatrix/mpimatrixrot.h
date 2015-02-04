/*
* Транспонирование матрицы
* Каждый процесс считывает свой диапазон колонок
* Транспонирует матрицу из этих колонок
* Сохраняет транспонированную матрицу из этих колонок
* Используются функции MPI_File_read_ordered и MPI_File_write_ordered
*/
#include <mpi.h>
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

	MPI_File file;
	MPI_Status status;
	mpiMatrixHeader header;
	int count;

	MPI_File_open(MPI_COMM_WORLD,inputFileName,MPI_MODE_RDONLY|MPI_MODE_SEQUENTIAL,MPI_INFO_NULL, &file);
	if(myrank==0) memset( &status, 0x00, sizeof(MPI_Status) );
	if(myrank==0) MPI_File_read_shared(file, &header, sizeof(mpiMatrixHeader), MPI_BYTE, &status);
	if(myrank==0) MPI_Get_count( &status, MPI_INT, &count );
	if(nrank>1) MPI_Bcast(&header, sizeof(mpiMatrixHeader), MPI_BYTE, 0, MPI_COMM_WORLD);

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

	// Переоткрываем файлы если требуется
	if(wrank<nrank) MPI_File_close(&file);
	if(myrank>=wrank) return;
	if(wrank<nrank) MPI_File_open(comm,inputFileName,MPI_MODE_RDONLY,MPI_INFO_NULL, &file);
	if(wrank<nrank) MPI_File_seek_shared(file,offset,MPI_SEEK_SET);

	int height2 = width1;
	int width2 = height1;

	int start = width1*myrank/wrank;
	int end = width1*(myrank+1)/wrank;
	int length=end-start;
	int bufferSize=length*height1;

	T *buffer1 = (T*)malloc(sizeof(T)*bufferSize);
	T *buffer2 = (T*)malloc(sizeof(T)*bufferSize);

	// Считываем построчно исходный файл
	// Каждый процесс считывает только свой диапазон колонок
	for(int i=0;i<height1;i++){
		memset( &status, 0x00, sizeof(MPI_Status) );
		MPI_File_read_ordered( file, &buffer1[i*length], length, dataType, &status);
		MPI_Get_count( &status, MPI_INT, &count );
		// assert(count==length);
	}

	MPI_File_close(&file);

	// Транспонируем колонки в строки
	for(int i=0;i<height1;i++){
		for(int j=0;j<length;j++){
			buffer2[j*width2+i]=buffer1[i*length+j];
		}
	}

	header.height=height2;
	header.width=width2;

	if(myrank==0) MPI_File_delete(outputFileName, MPI_INFO_NULL);
	MPI_File_open(comm,outputFileName,MPI_MODE_WRONLY|MPI_MODE_SEQUENTIAL|MPI_MODE_CREATE,MPI_INFO_NULL, &file);
	if(myrank==0) memset( &status, 0x00, sizeof(MPI_Status) );
	if(myrank==0) MPI_File_write_shared(file, &header, sizeof(mpiMatrixHeader), MPI_BYTE, &status);
	if(myrank==0) MPI_Get_count( &status, MPI_INT, &count );

	// Сохраняем строки
	memset( &status, 0x00, sizeof(MPI_Status) );
	MPI_File_write_ordered(file, buffer2, bufferSize, dataType, &status);
	MPI_Get_count( &status, MPI_INT, &count );

	MPI_File_close(&file);

	free(buffer1);
	free(buffer2);

	*counter += length*height1;
}

#endif
