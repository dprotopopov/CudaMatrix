/*
* Умножение матрицы на число (Multiply To Value)
* Число передаётся как первая матрица размера 1х1
* Каждый процесс читает и сохраняет свой диапазон ячеек
* Используется функции MPI_File_read_shared, MPI_File_read_ordered и MPI_File_write_ordered
*/
#include <mpi.h>
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

	MPI_File file1;
	MPI_File file2;
	MPI_Status status1;
	MPI_Status status2;
	mpiMatrixHeader header;
	int count1;
	int count2;
	T value;

	MPI_File_open(MPI_COMM_WORLD,inputFileName1,MPI_MODE_RDONLY|MPI_MODE_SEQUENTIAL,MPI_INFO_NULL, &file1);
	if(myrank==0) memset( &status1, 0x00, sizeof(MPI_Status) );
	if(myrank==0) MPI_File_read_shared(file1, &header, sizeof(mpiMatrixHeader), MPI_BYTE, &status1);
	if(myrank==0) MPI_Get_count( &status1, MPI_INT, &count1 );
	if(nrank>1) MPI_Bcast(&header, sizeof(mpiMatrixHeader), MPI_BYTE, 0, MPI_COMM_WORLD);

	assert(header.height==1);
	assert(header.width==1);
	assert(header.offset==sizeof(mpiMatrixHeader));

	MPI_Datatype dataType = header.dataType;
	MPI_Offset offset = header.offset;

	if(myrank==0) memset( &status1, 0x00, sizeof(MPI_Status) );
	if(myrank==0) MPI_File_read_shared(file1, &value, 1, dataType, &status1);
	if(myrank==0) MPI_Get_count( &status1, MPI_INT, &count1 );
	if(nrank>1) MPI_Bcast(&value, 1, dataType, 0, MPI_COMM_WORLD);
	MPI_File_close(&file1);

	MPI_File_open(MPI_COMM_WORLD,inputFileName2,MPI_MODE_RDONLY|MPI_MODE_SEQUENTIAL,MPI_INFO_NULL, &file1);
	if(myrank==0) memset( &status1, 0x00, sizeof(MPI_Status) );
	if(myrank==0) MPI_File_read_shared(file1, &header, sizeof(mpiMatrixHeader), MPI_BYTE, &status1);
	if(myrank==0) MPI_Get_count( &status1, MPI_INT, &count1 );
	if(nrank>1) MPI_Bcast(&header, sizeof(mpiMatrixHeader), MPI_BYTE, 0, MPI_COMM_WORLD);

	assert(header.dataType==dataType);
	assert(header.offset==sizeof(mpiMatrixHeader));

	int width = header.width;
	int height = header.height;

	int total = width*height;
	int wrank = min(nrank, total); // Количество реально используемых процессов

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
	if(wrank<nrank) MPI_File_close(&file1);
	if(myrank>=wrank) return;
	if(wrank<nrank) MPI_File_open(comm,inputFileName2,MPI_MODE_RDONLY,MPI_INFO_NULL, &file1);
	if(wrank<nrank) MPI_File_seek_shared(file1,offset,MPI_SEEK_SET);

	if(myrank==0) MPI_File_delete(outputFileName, MPI_INFO_NULL);
	MPI_File_open(comm,outputFileName,MPI_MODE_WRONLY|MPI_MODE_SEQUENTIAL|MPI_MODE_CREATE,MPI_INFO_NULL, &file2);

	if(myrank==0) memset( &status2, 0x00, sizeof(MPI_Status) );
	if(myrank==0) MPI_File_write_shared(file2, &header, sizeof(mpiMatrixHeader), MPI_BYTE, &status2);
	if(myrank==0) MPI_Get_count( &status2, MPI_INT, &count2 );

	int start = total*myrank/wrank;
	int end = total*(myrank+1)/wrank;
	int length=end-start;
	int bufferSize = length;

	T *buffer = (T*)malloc(sizeof(T)*bufferSize);

	for(int i=0;i<length;i+=bufferSize){
		int size = min(bufferSize, length-i);
		memset( &status1, 0x00, sizeof(MPI_Status) );
		memset( &status2, 0x00, sizeof(MPI_Status) );
		MPI_File_read_ordered( file1, buffer, size, dataType, &status1);
		for(int j=0;j<size;j++) buffer[j]*=value;
		MPI_File_write_ordered(file2, buffer, size, dataType, &status2);
		MPI_Get_count( &status1, MPI_INT, &count1 );
		MPI_Get_count( &status2, MPI_INT, &count2 );
		// assert(count1==size);
		// assert(count2==size);
	}

	free(buffer);

	MPI_File_close(&file1);
	MPI_File_close(&file2);

	*counter += length;
}

#endif
