/*
 * ������� �������
 * ������ ������� ��������� ���� �������� �����
 * ������������ ������� MPI_File_write_ordered
 */
#include <mpi.h>
#include <assert.h>
#include "mpimatrixfile.h"

#ifndef __MPIMATRIXNIL_H
#define __MPIMATRIXNIL_H

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
void mpi_matrix_nil(
	int height, int width, // ������� �������
	char *outputFileName, 
	long *counter) // ������� ���������� ��������
{
	int nrank;     /* ����� ���������� ��������� */
	int myrank;    /* ����� �������� �������� */

	MPI_Comm_size(MPI_COMM_WORLD, &nrank);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	int total = width*height;
	int wrank = min(nrank, total); // ���������� ������� ������������ ���������

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
	header.height = height;
	header.width = width;
	header.dataType = dataType;
	header.offset = sizeof(mpiMatrixHeader);

	if(myrank==0) MPI_File_delete(outputFileName, MPI_INFO_NULL);
	MPI_File_open(comm,outputFileName,MPI_MODE_WRONLY|MPI_MODE_SEQUENTIAL|MPI_MODE_CREATE,MPI_INFO_NULL, &file);

	MPI_Offset offset = header.offset;

	if(myrank==0) memset( &status, 0x00, sizeof(MPI_Status) );
	if(myrank==0) MPI_File_write_shared(file, &header, sizeof(mpiMatrixHeader), MPI_BYTE, &status);
	if(myrank==0) MPI_Get_count( &status, MPI_INT, &count );

	int start = total*myrank/wrank;
	int end = total*(myrank+1)/wrank;
	int length=end-start;
	int bufferSize = length;

	T nil = (T)0;
	T *buffer = (T*)malloc(sizeof(T)*bufferSize);

	for(int i=0;i<bufferSize;i++) buffer[i]=nil;

	for(int i=0;i<length;i+=bufferSize){
		int size = min(bufferSize, length-i);
		memset( &status, 0x00, sizeof(MPI_Status) );
		MPI_File_write_ordered(file, buffer, size, dataType, &status);
		MPI_Get_count( &status, MPI_INT, &count );
		// assert(count==size);
	}

	free(buffer);

	MPI_File_close(&file);

	*counter += length;
}

#endif
