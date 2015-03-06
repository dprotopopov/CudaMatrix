/*
* Единичная матрица
* Каждый процесс сохраняет свой диапазон ячеек
* Используется функция MPI_File_write_ordered, MPI_File_write
*/
#include <mpi.h>
#include <assert.h>
#include "mpimatrixfile.h"

#ifndef __MPIMATRIXONE_H
#define __MPIMATRIXONE_H

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
void mpi_matrix_one(
	int rank, 
	char *outputFileName, 
	long *counter) // Счётчик количества операций
{
	int nrank;     /* Общее количество процессов */
	int myrank;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &nrank);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	int wrank = min(nrank, rank); // Количество реально используемых процессов

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

	MPI_File file;
	MPI_Status status;
	mpiMatrixHeader header;
	int count;
	MPI_Datatype dataType = MPI_DOUBLE;

	header.fourCC[0] = 'M';
	header.fourCC[1] = 'P';
	header.fourCC[2] = 'I';
	header.fourCC[3] = 'M';
	header.height = rank;
	header.width = rank;
	header.dataType = dataType;
	header.offset = sizeof(mpiMatrixHeader);

	if(myrank==0) MPI_File_delete(outputFileName, MPI_INFO_NULL);
	MPI_File_open(comm,outputFileName,MPI_MODE_WRONLY|MPI_MODE_SEQUENTIAL|MPI_MODE_CREATE,MPI_INFO_NULL, &file);

	MPI_Offset offset = header.offset;

	if(myrank==0) memset( &status, 0x00, sizeof(MPI_Status) );
	if(myrank==0) MPI_File_write_shared(file, &header, sizeof(mpiMatrixHeader), MPI_BYTE, &status);
	if(myrank==0) MPI_Get_count( &status, MPI_INT, &count );

	int start = rank*myrank/wrank;
	int end = rank*(myrank+1)/wrank;
	int length=end-start;
	if(myrank==0) length++;

	T nil = (T)0;
	T *buffer = (T*)malloc(sizeof(T)*length+1);

	for(int i=0;i<length;i++) buffer[i]=nil;
	if(myrank==0) buffer[0]=(T)1;

	for(int i=0;i<rank-1;i++){
		memset( &status, 0x00, sizeof(MPI_Status) );
		MPI_File_write_ordered(file, buffer, length, dataType, &status);
		MPI_Get_count( &status, MPI_INT, &count );
		// assert(count==length);
	}
	if(myrank==0) memset( &status, 0x00, sizeof(MPI_Status) );
	if(myrank==0) MPI_File_write_shared( file, buffer, 1, dataType, &status);
	if(myrank==0) MPI_Get_count( &status, MPI_INT, &count );
	if(wrank>1) MPI_Barrier( comm );
	MPI_File_close(&file);

	free(buffer);

	*counter += length*(rank-1);
	if(myrank==0) *counter+=1;
}

#endif
