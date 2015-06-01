/*
* ���������� ������� � ������������� ���� ���������������� ����� ������� ������-�������
* ������ ������� ������������ ���� �������� �������
* � �������������� ������� ��������� �������������� ��� � � ��������
* ������ ���������� ������ ������-������� �� ��������� �������� ������� � �����������
* ��������� �������������� ������������ ����� �������� �������� ������ ��������� ��������� � ������
* ��� ���������� ������������ ��������� �������� ��������� ������������ ������ ��������� ��������� 
* �� ���� �������� �������������� ������������ �����
*/
#include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include "mpimatrixfile.h"
#include "minmax.h"

#ifndef __MPIMATRIXGAUSSJORDAN_H
#define __MPIMATRIXGAUSSJORDAN_H

template <typename T>
void mpi_matrix_gaussjordan_step(
	MPI_Comm comm, // ������������ ������ ������������ ���������
	T *buffer, // ��������� ����� �������� �������� �������
	T *buffer2, // ��������� ����� �������� �������������� �������
	T *columnBuffer, // ����������������� ����� ��� �������� �������� ������� �������� �������
	T *totalDiv, // ������������ ������ ��������� ��������� ���������� �������� �������
	int row, int col, // ������ � ������� �������� �������� � ��������� ������-�������
	int height, int width, int length2, // ������� �������� ������� � ��������� ������ �������������� �������
	MPI_Datatype dataType, // ��� ��� ������ �������
	int *bounds, // ������� ������� �������� �������� ������� �� ���������
	int mp, int wrank, // ����� �������� � ���������� ������������ ���������
	long *counter) // ������� ���������� ��������
{
	int start = bounds[mp];
	int end = bounds[mp+1];
	int length=end-start;

	// ����������� �������� ������� �������� ������� � ����������� ��� ����� ����� ����������
	if (start<=col && col<end) {
		int j=col-start;
		for(int i=0;i<height;i++){
			columnBuffer[i]=buffer[i*length+j];
		}
	}

	int root = wrank;
	while(root-->0) if(bounds[root]<=col && col<bounds[root+1]) break;
	if(wrank>1) MPI_Bcast(columnBuffer, height, dataType, root, comm);

	T d=columnBuffer[row]; // �������� �������� ��������
	assert(d!=(T)0);

	for(int id=0;id<length*height;id++){
		int i=id/length;
		int j=id%length;
		if(start+j==col) continue;
		if(i==row) continue;
		T a = columnBuffer[i];
		T b = buffer[row*length+j];
		buffer[id]-=a*b/d;
	}
	for(int j=0; j<length; j++) {
		buffer[row*length+j]/=d;
	}
	if (start<=col && col<end) {
		int j=col-start;
		for(int i=0;i<height;i++){
			buffer[i*length+j]=(T)0;
		}
		buffer[row*length+j]=(T)1;
	}
	for(int id=0;id<length2*height;id++){
		int i=id/length2;
		int j=id%length2;
		if(i==row) continue;
		T a = columnBuffer[i];
		T b = buffer2[row*length2+j];
		buffer2[id]-=a*b/d;
	}
	for(int j=0; j<length2; j++) {
		buffer2[row*length2+j]/=d;
	}

	*counter += (length+length2)*height;
	*totalDiv *= d;
}

/*
* ���������� �������� ������ ��������� ��������� � ������� �������
* ������ ������� ������������ ���� �������� �������
* ������������ MPI_Allreduce
*/
template <typename T>
void mpi_matrix_indexOfFirstNotZero(
	MPI_Comm comm, // ������������ ������ ������������ ���������
	T *buffer, // ��������� ����� �������� �������� �������
	int *localIndex, // ����������������� ����� ��� �������� �������� ��������� ���������
	int *totalIndex, // ����������������� ����� ��� �������� �������� ��������� ���������
	int height, int width, // ������� �������
	MPI_Datatype dataType, // ��� ��� ������ �������
	int mp, int wrank, // ����� �������� � ���������� ������������ ���������
	long *counter) // ������� ���������� ��������
{
	int start = width*mp/wrank;
	int end = width*(mp+1)/wrank;
	int length=end-start;

	// ������� ������� ������� ��������� ���������
	for(int i=0;i<height;i++) {
		localIndex[i] = width;
		for(int j=0;j<length;j++){
			if(buffer[i*length+j]!=(T)0){
				localIndex[i]=start+j;
				break;
			}
		}
	}

	if(wrank>1) MPI_Allreduce(localIndex,totalIndex,height,MPI_INT,MPI_MIN,comm);
	else memcpy(totalIndex,localIndex,sizeof(T)*height);
}

template <typename T>
void mpi_matrix_gaussjordan(
	MPI_Comm comm, // ������������ ������ ������������ ���������
	T *buffer, // ��������� ����� �������� �������� �������
	T *buffer2, // ��������� ����� �������� �������������� �������
	T *totalDiv, // ������������ ������ ��������� ��������� ���������� �������� �������
	int *localIndex, // ����������������� ����� ��� �������� �������� ��������� ���������
	int *totalIndex, // ����������������� ����� ��� �������� �������� ��������� ���������
	int height, int width, int length2, // ������� �������� ������� � ��������� ������ �������������� �������
	MPI_Datatype dataType, // ��� ��� ������ �������
	int mp, int wrank, // ����� �������� � ���������� ������������ ���������
	long *counter) // ������� ���������� ��������
{
	T *columnBuffer = (T*)malloc(sizeof(T)*height+1);
	int *bounds = (int *)malloc(sizeof(int)*(wrank+1));
	for(int i=0;i<=wrank;i++) bounds[i] = width*i/wrank;
	int start = bounds[mp];
	int end = bounds[mp+1];
	int length=end-start;
	int start1 = height*mp/wrank;
	int end1 = height*(mp+1)/wrank;
	int length1=end1-start1;

	*totalDiv = (T)1;

	for(int col=0;col<width;col++) {

		// ������� ������� ������� ��������� ���������
		mpi_matrix_indexOfFirstNotZero<T>(
			comm, // ������������ ������ ������������ ���������
			buffer, 
			localIndex, 
			totalIndex, 
			height, width, 
			dataType, 
			mp, wrank, 
			counter);

		// ������� ��������� ����������� ������ ���������� ��������
		int localIndexValue = width;
		int totalIndexValue = width;

		for(int i=start1;i<end1;i++) {
			if(col<=totalIndex[i]&&totalIndex[i]<=localIndexValue) {
				localIndexValue=totalIndex[i];
			}
		}

		if(wrank>1) MPI_Allreduce(&localIndexValue,&totalIndexValue,1,MPI_INT,MPI_MIN,comm);
		else totalIndexValue=localIndexValue;

		if (totalIndexValue==width) {
			*totalDiv = (T)0;
			break;
		}

		col=totalIndexValue;
		int row = height;
		while(row-->0) if(totalIndex[row]==totalIndexValue) break;

		mpi_matrix_gaussjordan_step<T>(
			comm, // ������������ ������ ������������ ���������
			buffer, // ��������� ����� �������� �������� �������
			buffer2, // ��������� ����� �������� �������������� �������
			columnBuffer, // ����������������� ����� ��� �������� �������� ������� �������� �������
			totalDiv, // ������������ ������ ��������� ��������� ���������� �������� �������
			row, col, // ������ � ������� �������� �������� � ��������� ������-�������
			height, width, length2,// ������� �������� ������� � ��������� ������ �������������� �������
			dataType, // ��� ��� ������ �������
			bounds, // ������� ������� �������� �������� ������� �� ���������
			mp, wrank, // ����� �������� � ���������� ������������ ���������
			counter); // ������� ���������� ��������
	}

	free(bounds);
	free(columnBuffer);
}
#endif
