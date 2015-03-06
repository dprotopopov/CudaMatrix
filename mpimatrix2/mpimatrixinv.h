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
* ������������ ������� fread fwrite
*/
#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include "mpimatrixfile.h"
#include "mpimatrixgaussjordan.h"

#ifndef __MPIMATRIXINV_H
#define __MPIMATRIXINV_H

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

	FILE *file=NULL;
	mpiMatrixHeader header;

	if(myrank==0) file = fopen(inputFileName,"rb");
	if(myrank==0) fread(&header,1,sizeof(mpiMatrixHeader),file);
	if(nrank>1) MPI_Bcast(&header,sizeof(mpiMatrixHeader),MPI_BYTE,0,MPI_COMM_WORLD);

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

	if(myrank>=wrank) return;

	int start = rank*myrank/wrank;
	int end = rank*(myrank+1)/wrank;
	int length=end-start;
	int bufferSize=rank;
	int bufferSize1=length*rank;
	int bufferSize2=length*rank;

	T *buffer = (T*)malloc(sizeof(T)*bufferSize+1);
	T *buffer1 = (T*)malloc(sizeof(T)*bufferSize1+1);
	T *buffer2 = (T*)malloc(sizeof(T)*bufferSize2+1);

	// ��������� ��������� �������� ����
	// ������ ������� ��������� ������ ���� �������� �������
	for(int i=0;i<rank;i++){
		if(myrank==0) fread(buffer,sizeof(T),rank,file);
		for(int j=0;j<wrank;j++) {
			int start = rank*j/wrank;
			int end = rank*(j+1)/wrank;
			int count=end-start;
			if(j==0 && myrank==0) memcpy(&buffer1[i*length],buffer,length*sizeof(T));
			else if(myrank==0) MPI_Send(&buffer[start],count,dataType,j,0,comm);
			else if(myrank==j) MPI_Recv(&buffer1[i*length],length,dataType,0,0,comm,MPI_STATUS_IGNORE);
		}
	}

	if(myrank==0) fclose(file);

	T nil = (T)0;
	T one = (T)1;
	for(int i=0;i<bufferSize2;i++) buffer2[i]=nil;
	
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
	int *ii = (int *)malloc(sizeof(int)*rank);
	for(int i=0;i<rank;i++){
		ii[totalIndex[i]]=i;
	}

	if(myrank==0) file = fopen(outputFileName,"wb");
	if(myrank==0) fwrite(&header,1,sizeof(mpiMatrixHeader),file);

	// ��������� ���������
	// ������ ������� ��������� ������ ���� �������� �������
	// ������� ����� ��� ���������� ������������ ��������� ������ ��������� ���������
	for(int i=0;i<rank;i++){
		for(int j=0;j<wrank;j++) {
			int start = rank*j/wrank;
			int end = rank*(j+1)/wrank;
			int count=end-start;
			if(j==0 && myrank==0) memcpy(buffer,&buffer2[ii[i]*length],length*sizeof(T));
			else if(myrank==0) MPI_Recv(&buffer[start],count,dataType,j,0,comm,MPI_STATUS_IGNORE);
			else if(myrank==j) MPI_Send(&buffer2[ii[i]*length],length,dataType,0,0,comm);
		}
		if(myrank==0) fwrite(buffer,sizeof(T),rank,file);
	}

	if(myrank==0) fclose(file);

	free(ii);
	free(localIndex);
	free(totalIndex);
	free(buffer);
	free(buffer1);
	free(buffer2);
}

#endif
