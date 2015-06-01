/*
* Умножение матрицы
* Каждый процесс считывает свой диапазон колонок второй матрицы
* Читаем построчно первую матрицу и умножаем на колонки второй матрицы
* Сохраняем полученый вектор
* Используются функции MPI_File_read_ordered, MPI_File_read_shared и MPI_File_write_ordered
*/
#include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include "mpimatrixfile.h"
#include "minmax.h"

#ifndef __MPIMATRIXMUL_H
#define __MPIMATRIXMUL_H

template <typename T>
void mpi_matrix_mul(
	char *inputFileName1, 
	char *inputFileName2, 
	char *outputFileName, 
	long *counter) // Счётчик количества операций
{
	int np;    /* Общее количество процессов */
	int mp;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &mp);

	MPI_File file1;
	MPI_File file2;
	MPI_Status status1;
	MPI_Status status2;
	mpiMatrixHeader header1;
	mpiMatrixHeader header2;
	int count1;
	int count2;

	MPI_File_open(MPI_COMM_WORLD,inputFileName2,MPI_MODE_RDONLY|MPI_MODE_SEQUENTIAL,MPI_INFO_NULL, &file2);
	if(mp==0) memset( &status2, 0x00, sizeof(MPI_Status) );
	if(mp==0) MPI_File_read_shared(file2, &header2, sizeof(mpiMatrixHeader), MPI_BYTE, &status2);
	if(mp==0) MPI_Get_count( &status2, MPI_INT, &count2 );
	if(np>1) MPI_Bcast(&header2, sizeof(mpiMatrixHeader), MPI_BYTE, 0, MPI_COMM_WORLD);

	int height2 = header2.height;
	int width2 = header2.width;

	MPI_Datatype dataType = header2.dataType;
	MPI_Offset offset = header2.offset;

	int wrank = min(np, width2); // Количество реально используемых процессов

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
	if(wrank<np) MPI_File_close(&file2);
	if(mp>=wrank) return;
	if(wrank<np) MPI_File_open(comm,inputFileName2,MPI_MODE_RDONLY,MPI_INFO_NULL, &file2);
	if(wrank<np) MPI_File_seek_shared(file2,offset,MPI_SEEK_SET);

	int start = width2*mp/wrank;
	int end = width2*(mp+1)/wrank;
	int length=end-start;
	int bufferSize=length*height2;

	T *buffer2 = (T*)malloc(sizeof(T)*bufferSize);

	// Считываем построчно исходный файл
	// Каждый процесс считывает только свой диапазон колонок
	for(int i=0;i<height2;i++){
		memset( &status2, 0x00, sizeof(MPI_Status) );
		MPI_File_read_ordered( file2, &buffer2[i*length], length, dataType, &status2);
		MPI_Get_count( &status2, MPI_INT, &count2 );
		// assert(count2==length);
	}

	MPI_File_close(&file2);

	MPI_File_open(comm,inputFileName1,MPI_MODE_RDONLY|MPI_MODE_SEQUENTIAL,MPI_INFO_NULL, &file1);
	if(mp==0) memset( &status1, 0x00, sizeof(MPI_Status) );
	if(mp==0) MPI_File_read_shared(file1, &header1, sizeof(mpiMatrixHeader), MPI_BYTE, &status1);
	if(mp==0) MPI_Get_count( &status1, MPI_INT, &count1 );
	if(np>1) MPI_Bcast(&header1, sizeof(mpiMatrixHeader), MPI_BYTE, 0, MPI_COMM_WORLD);

	assert(header1.dataType==header2.dataType);
	assert(header1.width==header2.height);
	assert(header1.offset==header2.offset);
	assert(header1.offset==sizeof(mpiMatrixHeader));
	assert(header2.offset==sizeof(mpiMatrixHeader));

	int height1 = header1.height;
	int width1 = header1.width;
	header2.height = height1;

	T *buffer1 = (T*)malloc(sizeof(T)*width1+1);
	T *buffer3 = (T*)malloc(sizeof(T)*length+1);

	if(mp==0) MPI_File_delete(outputFileName, MPI_INFO_NULL);
	MPI_File_open(comm,outputFileName,MPI_MODE_WRONLY|MPI_MODE_SEQUENTIAL|MPI_MODE_CREATE,MPI_INFO_NULL, &file2);

	if(mp==0) memset( &status2, 0x00, sizeof(MPI_Status) );
	if(mp==0) MPI_File_write_shared(file2, &header2, sizeof(mpiMatrixHeader), MPI_BYTE, &status2);
	if(mp==0) MPI_Get_count( &status2, MPI_INT, &count2 );

	// Считываем построчно исходный файл
	// Все процессы считывают одинаковые данные
	// Умножаем строку на столбцы в памяти
	// Сохраняем построчно
	// Каждый процесс сохраняет только свой диапазон колонок
	for(int i=0;i<height1;i++){
		if(mp==0) memset( &status1, 0x00, sizeof(MPI_Status) );
		if(mp==0) MPI_File_read_shared(file1, buffer1, width1, dataType, &status1);
		if(np>1) MPI_Bcast(buffer1, width1, dataType, 0, MPI_COMM_WORLD);
		for(int j=0;j<length;j++){
			T s = (T)0;
			for(int k=0;k<width1;k++){
				s+=buffer1[k]*buffer2[k*length+j];
			}
			buffer3[j]=s;
		}
		memset( &status2, 0x00, sizeof(MPI_Status) );
		MPI_File_write_ordered(file2, buffer3, length, dataType, &status2);
		MPI_Get_count( &status2, MPI_INT, &count2 );
		// assert(count2==length);
	}

	MPI_File_close(&file1);
	MPI_File_close(&file2);

	free(buffer1);
	free(buffer2);
	free(buffer3);

	*counter += height1*width1*length;
}

#endif
