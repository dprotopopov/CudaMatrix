/*
* ����������� �������
* ������ ������� ������ � ��������� ���� �������� �����
* ������������ ������� MPI_File_read_ordered � MPI_File_write_ordered
*/
#include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include "mpimatrixfile.h"
#include "minmax.h"

#ifndef __MPIMATRIXMOV_H
#define __MPIMATRIXMOV_H

template <typename T>
void mpi_matrix_mov(
	char *inputFileName, 
	char *outputFileName, 
	long *counter) // ������� ���������� ��������
{
	int np;    /* ����� ���������� ��������� */
	int mp;    /* ����� �������� �������� */

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &mp);

	MPI_File file1;
	MPI_File file2;
	MPI_Status status1;
	MPI_Status status2;
	mpiMatrixHeader header;
	int count1;
	int count2;

	MPI_File_open(MPI_COMM_WORLD,inputFileName,MPI_MODE_RDONLY|MPI_MODE_SEQUENTIAL,MPI_INFO_NULL, &file1);

	if(mp==0) memset( &status1, 0x00, sizeof(MPI_Status) );
	if(mp==0) MPI_File_read_shared(file1, &header, sizeof(mpiMatrixHeader), MPI_BYTE, &status1);
	if(mp==0) MPI_Get_count( &status1, MPI_INT, &count1 );
	if(np>1) MPI_Bcast(&header, sizeof(mpiMatrixHeader), MPI_BYTE, 0, MPI_COMM_WORLD);

	assert(header.offset==sizeof(mpiMatrixHeader));

	int width = header.width;
	int height = header.height;
	MPI_Datatype dataType = header.dataType;
	MPI_Offset offset = header.offset;

	int total = width*height;
	int wrank = min(np, total); // ���������� ������� ������������ ���������

	// ������ ������ �� ������� ������������ ��������� � ������������ ���� ������
	MPI_Group world_group;
	MPI_Group group;
	MPI_Comm comm;
	int *ranks = (int *)malloc(sizeof(int)*wrank);
	for(int i=0;i<wrank;i++) ranks[i]=i;
	MPI_Comm_group(MPI_COMM_WORLD,&world_group);
	MPI_Group_incl(world_group,wrank,ranks,&group);
	MPI_Comm_create(MPI_COMM_WORLD,group,&comm);
	free(ranks);

	// ������������� ����� ���� ���������
	if(wrank<np) MPI_File_close(&file1);
	if(mp>=wrank) return;
	if(wrank<np) MPI_File_open(comm,inputFileName,MPI_MODE_RDONLY,MPI_INFO_NULL, &file1);
	if(wrank<np) MPI_File_seek_shared(file1,offset,MPI_SEEK_SET);

	if(mp==0) MPI_File_delete(outputFileName, MPI_INFO_NULL);
	MPI_File_open(comm,outputFileName,MPI_MODE_WRONLY|MPI_MODE_SEQUENTIAL|MPI_MODE_CREATE,MPI_INFO_NULL, &file2);

	if(mp==0) memset( &status2, 0x00, sizeof(MPI_Status) );
	if(mp==0) MPI_File_write_shared(file2, &header, sizeof(mpiMatrixHeader), MPI_BYTE, &status2);
	if(mp==0) MPI_Get_count( &status2, MPI_INT, &count2 );

	int start = total*mp/wrank;
	int end = total*(mp+1)/wrank;
	int length=end-start;
	int bufferSize = length;

	T *buffer = (T*)malloc(sizeof(T)*bufferSize);

	for(int i=0;i<length;i+=bufferSize){
		int size = min(bufferSize, length-i);
		memset( &status1, 0x00, sizeof(MPI_Status) );
		memset( &status2, 0x00, sizeof(MPI_Status) );
		MPI_File_read_ordered( file1, buffer, size, dataType, &status1);
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
