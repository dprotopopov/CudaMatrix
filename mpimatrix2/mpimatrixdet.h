/*
* ���������� ������������ �������
* ������ ������� ��������� ���� �������� �������
* �������� ������� ���������� ���������������� ������-������� � ������������� ����
* ������ ���������� ������ ������-������� �� ��������� �������� ������� � �����������
* ��������� �������������� ������������ ����� �������� �������� ������ ��������� ��������� � ������
* ��� ���������� ������������ ��������� �������� ��������� ������������ ������ ��������� ��������� 
* �� ���� �������� �������������� ������������ �����
* ����������� ������������ ������� ��� ������� 1�1
* ������������ ������� fread fwrite
*/
#include <mpi.h>
#include <stdio.h>
#include <assert.h>
#include "mpimatrixfile.h"
#include "mpimatrixgaussjordan.h"

#ifndef __MPIMATRIXDET_H
#define __MPIMATRIXDET_H

template <typename T>
void mpi_matrix_det(
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

	T *buffer = (T*)malloc(sizeof(T)*bufferSize);
	T *buffer1 = (T*)malloc(sizeof(T)*bufferSize1);

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
	(T *)NULL, // ��������� ����� �������� �������������� �������
	&totalDiv, // ������������ ������ ��������� ��������� ���������� �������� �������
	localIndex, // ����������������� ����� ��� �������� �������� ��������� ���������
	totalIndex, // ����������������� ����� ��� �������� �������� ��������� ���������
	rank, rank, 0, // ������� �������� ������� � ��������� ������ �������������� �������
	dataType, // ��� ��� ������ �������
	myrank, wrank, // ����� �������� � ���������� ���������
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
	myrank, wrank, // ����� �������� � ���������� ���������
	counter); // ������� ���������� ��������

	if(totalDiv!=(T)0) {
		// ����������� �������� ������������
		// ���� ������������ - �� ������ ��������� ���������� ������������ � ����������� ����������
		long count = 0;
		for(int i=0;i<rank-1;i++){
			for(int j=i+1;j<rank; j++){
				int a = totalIndex[i];
				int b = totalIndex[j];
				if(a>b){
					totalIndex[i]=b;
					totalIndex[j]=a;
					count++;
				}
			}
		}
		if(count&1) totalDiv=-totalDiv;
	}

	header.height=1;
	header.width=1;

	if(myrank==0) file = fopen(outputFileName,"wb");
	if(myrank==0) fwrite(&header,1,sizeof(mpiMatrixHeader),file);
	if(myrank==0) fwrite(&totalDiv,sizeof(T),1,file);
	if(myrank==0) fclose(file);

	free(localIndex);
	free(totalIndex);
	free(buffer);
	free(buffer1);
}

#endif
