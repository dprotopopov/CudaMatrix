/*
* ���������� �������� �������
* ������ ������� ��������� ���� �������� �������
* ������������� �������������� ��������� �������
* �������� ������� ���������� ���������������� ������-������� � ������������� ����
* � �������������� ������� ��������� �������������� ��� � � ��������
* ������ ���������� ������ ������-������� �� ��������� �������� ������� � �����������
* ��������� �������������� ������������ ����� �������� �������� ������ ��������� ��������� � ������
* ��� ���������� ������������ ��������� �������� ��������� ������������ ������ ��������� ��������� 
* �� ���� �������� �������������� ������������ �����
* ��������� �������� ������� �� ���� �������
* ������������ ������� MPI_File_read_ordered � MPI_File_write_ordered
*/
#include <mpi.h>
#include <stdlib.h>     /* qsort */
#include <assert.h>
#include "mpimatrixfile.h"
#include "mpimatrixgaussjordan.h"

#ifndef __MPIMATRIXINV_H
#define __MPIMATRIXINV_H

#ifndef INDEX_INDEX
#define INDEX_INDEX

typedef struct {
	int index1;
	int index2;
} index_index;

int compare_index_index (const void * a, const void * b)
{
  return ( ((index_index*)a)->index1 - ((index_index*)b)->index1 );
}

#endif // INDEX_INDEX

template <typename T>
void mpi_matrix_inv(
	char *inputFileName, 
	char *outputFileName, 
	long *counter) // ������� ���������� ��������
{
	int nrank;     /* ����� ���������� ��������� */
	int myrank;    /* ����� �������� �������� */

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

	assert(header.height==header.width);
	assert(header.offset==sizeof(mpiMatrixHeader));

	MPI_Datatype dataType = header.dataType;
	MPI_Offset offset = header.offset;

	int rank = header.height;
	int wrank = min(nrank, rank); // ���������� ������� ������������ ���������

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
	if(wrank<nrank) MPI_File_close(&file);
	if(myrank>=wrank) return;
	if(wrank<nrank) MPI_File_open(comm,inputFileName,MPI_MODE_RDONLY,MPI_INFO_NULL, &file);
	if(wrank<nrank) MPI_File_seek_shared(file,offset,MPI_SEEK_SET);

	int start = rank*myrank/wrank;
	int end = rank*(myrank+1)/wrank;
	int length=end-start;
	int bufferSize=length*rank;

	T *buffer1 = (T*)malloc(sizeof(T)*bufferSize);
	T *buffer2 = (T*)malloc(sizeof(T)*bufferSize);

	// ��������� ��������� �������� ����
	// ������ ������� ��������� ������ ���� �������� �������
	for(int i=0;i<rank;i++){
		memset( &status, 0x00, sizeof(MPI_Status) );
		MPI_File_read_ordered( file, &buffer1[i*length], length, dataType, &status);
		MPI_Get_count( &status, MPI_INT, &count );
		// assert(count==length);
	}

	MPI_File_close(&file);

	T nil = (T)0;
	T one = (T)1;
	for(int j=0;j<bufferSize;j++) buffer2[j]=nil;
	
	for(int i=0;i<length;i++){
		buffer2[(start+i)*length+i]=one;
	}

	T totalDiv;
	int *localIndex = (int *)malloc(sizeof(int)*rank);
	int *totalIndex = (int *)malloc(sizeof(int)*rank);

	/*
	* ���������� ������� � ������������� ���� ���������������� ����� ������� ������-�������
	* ������ ������� ������������ ���� �������� �������
	* � �������������� ������� ��������� �������������� ��� � � ��������
	* ������ ���������� ������ ������-������� �� ��������� �������� ������� � �����������
	* ��������� �������������� ������������ ����� �������� �������� ������ ��������� ��������� � ������
	* ��� ���������� ������������ ��������� �������� ��������� ������������ ������ ��������� ��������� 
	* �� ���� �������� �������������� ������������ �����
	*/
	mpi_matrix_gaussjordan(
	comm, // ������������ ������ ������������ ���������
	buffer1, // ��������� ����� �������� �������� �������
	buffer2, // ��������� ����� �������� �������������� �������
	&totalDiv, // ������������ ������ ��������� ��������� ���������� �������� �������
	localIndex, // ����������������� ����� ��� �������� �������� ��������� ���������
	totalIndex, // ����������������� ����� ��� �������� �������� ��������� ���������
	rank, rank, length, // ������� �������� ������� � ��������� ������ �������������� �������
	dataType, // ��� ��� ������ �������
	myrank, wrank, // ����� �������� � ���������� ������������ ���������
	counter); // ������� ���������� ��������

	/*
	* ���������� �������� ������ ��������� ��������� � ������� �������
	* ������ ������� ������������ ���� �������� �������
	* ������������ MPI_Allreduce
	*/
	mpi_matrix_indexOfFirstNotZero(
	comm, // ������������ ������ ������������ ���������
	buffer1, // ��������� ����� �������� �������� �������
	localIndex, // ����������������� ����� ��� �������� �������� ��������� ���������
	totalIndex, // ����������������� ����� ��� �������� �������� ��������� ���������
	rank, rank, // ������� �������
	dataType, // ��� ��� ������ �������
	myrank, wrank, // ����� �������� � ���������� ������������ ���������
	counter); // ������� ���������� ��������

	// ���������� ��������� ������������ �����
	index_index *ii = (index_index *)malloc(sizeof(index_index)*rank);
	for(int i=0;i<rank;i++){
		ii[i].index1=totalIndex[i];
		ii[i].index2=i;
	}

	qsort (ii, rank, sizeof(index_index), compare_index_index);

	if(myrank==0) MPI_File_delete(outputFileName, MPI_INFO_NULL);
	MPI_File_open(comm,outputFileName,MPI_MODE_WRONLY|MPI_MODE_SEQUENTIAL|MPI_MODE_CREATE,MPI_INFO_NULL, &file);

	if(myrank==0) memset( &status, 0x00, sizeof(MPI_Status) );
	if(myrank==0) MPI_File_write_shared(file, &header, sizeof(mpiMatrixHeader), MPI_BYTE, &status);

	// ��������� ���������
	// ������ ������� ��������� ������ ���� �������� �������
	// ������� ����� ��� ���������� ������������ ��������� ������ ��������� ���������
	for(int i=0;i<rank;i++){
		memset( &status, 0x00, sizeof(MPI_Status) );
		MPI_File_write_ordered(file, &buffer2[ii[i].index2*length], length, dataType, &status);
		MPI_Get_count( &status, MPI_INT, &count );
		// assert(count==length);
	}

	MPI_File_close(&file);

	free(ii);
	free(localIndex);
	free(totalIndex);
	free(buffer1);
	free(buffer2);
}

#endif
