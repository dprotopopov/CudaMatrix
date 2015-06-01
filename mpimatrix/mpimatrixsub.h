/*
* Вычитание матриц
* Каждый процесс считывает свой диапазон ячеек
* Производит вычитание ячеек
* Сохраняет вычисленные ячейки
* Используются функции MPI_File_read_ordered и MPI_File_write_ordered
*/
#include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include "mpimatrixfile.h"
#include "minmax.h"

#ifndef __MPIMATRIXSUB_H
#define __MPIMATRIXSUB_H

template <typename T>
void mpi_matrix_sub(
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
	MPI_File file3;
	MPI_Status status1;
	MPI_Status status2;
	MPI_Status status3;
	mpiMatrixHeader header1;
	mpiMatrixHeader header2;
	int count1;
	int count2;
	int count3;

	MPI_File_open(MPI_COMM_WORLD,inputFileName1,MPI_MODE_RDONLY|MPI_MODE_SEQUENTIAL,MPI_INFO_NULL, &file1);
	MPI_File_open(MPI_COMM_WORLD,inputFileName2,MPI_MODE_RDONLY|MPI_MODE_SEQUENTIAL,MPI_INFO_NULL, &file2);

	if(mp==0) memset( &status1, 0x00, sizeof(MPI_Status) );
	if(mp==0) memset( &status2, 0x00, sizeof(MPI_Status) );
	if(mp==0) MPI_File_read_shared(file1, &header1, sizeof(mpiMatrixHeader), MPI_BYTE, &status1);
	if(mp==0) MPI_File_read_shared(file2, &header2, sizeof(mpiMatrixHeader), MPI_BYTE, &status2);
	if(mp==0) MPI_Get_count( &status1, MPI_INT, &count1 );
	if(mp==0) MPI_Get_count( &status2, MPI_INT, &count2 );
	if(np>1) MPI_Bcast(&header1, sizeof(mpiMatrixHeader), MPI_BYTE, 0, MPI_COMM_WORLD);
	if(np>1) MPI_Bcast(&header2, sizeof(mpiMatrixHeader), MPI_BYTE, 0, MPI_COMM_WORLD);

	assert(header1.dataType==header2.dataType);
	assert(header1.width==header2.width);
	assert(header1.height==header2.height);
	assert(header1.offset==header2.offset);
	assert(header1.offset==sizeof(mpiMatrixHeader));
	assert(header2.offset==sizeof(mpiMatrixHeader));

	int width = header1.width;
	int height = header1.height;

	MPI_Datatype dataType = header1.dataType;
	MPI_Offset offset = header1.offset;
	int total = width*height;

	int wrank = min(np, total); // Количество реально используемых процессов

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
	if(wrank<np) MPI_File_close(&file1);
	if(wrank<np) MPI_File_close(&file2);
	if(mp>wrank) return;
	if(wrank<np) MPI_File_open(comm,inputFileName1,MPI_MODE_RDONLY|MPI_MODE_SEQUENTIAL,MPI_INFO_NULL, &file1);
	if(wrank<np) MPI_File_open(comm,inputFileName2,MPI_MODE_RDONLY|MPI_MODE_SEQUENTIAL,MPI_INFO_NULL, &file2);
	if(wrank<np) MPI_File_seek_shared(file1,offset,MPI_SEEK_SET);
	if(wrank<np) MPI_File_seek_shared(file2,offset,MPI_SEEK_SET);

	if(mp==0) MPI_File_delete(outputFileName, MPI_INFO_NULL);
	MPI_File_open(comm,outputFileName,MPI_MODE_WRONLY|MPI_MODE_SEQUENTIAL|MPI_MODE_CREATE,MPI_INFO_NULL, &file3);

	if(mp==0) memset( &status3, 0x00, sizeof(MPI_Status) );
	if(mp==0) MPI_File_write_shared(file3, &header1, sizeof(mpiMatrixHeader), MPI_BYTE, &status3);
	if(mp==0) MPI_Get_count( &status3, MPI_INT, &count3 );

	int start = total*mp/wrank;
	int end = total*(mp+1)/wrank;
	int length=end-start;
	int bufferSize = length;

	T *buffer1 = (T*)malloc(sizeof(T)*bufferSize);
	T *buffer2 = (T*)malloc(sizeof(T)*bufferSize);

	for(int i=0;i<length;i+=bufferSize){
		int size = min(bufferSize, length-i);
		memset( &status1, 0x00, sizeof(MPI_Status) );
		memset( &status2, 0x00, sizeof(MPI_Status) );
		memset( &status3, 0x00, sizeof(MPI_Status) );
		MPI_File_read_ordered( file1, buffer1, size, dataType, &status1);
		MPI_File_read_ordered( file2, buffer2, size, dataType, &status2);
		for(int j=0;j<size;j++) buffer1[j]-=buffer2[j];
		MPI_File_write_ordered(file3, buffer1, size, dataType, &status3);
		MPI_Get_count( &status1, MPI_INT, &count1 );
		MPI_Get_count( &status2, MPI_INT, &count2 );
		MPI_Get_count( &status3, MPI_INT, &count3 );
		// assert(count1==size);
		// assert(count2==size);
		// assert(count3==size);
	}

	free(buffer1);
	free(buffer2);
	MPI_File_close(&file1);
	MPI_File_close(&file2);
	MPI_File_close(&file3);

	*counter += length;
}

#endif
