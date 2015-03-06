#include <mpi.h>

#ifndef __MPIMATRIXFILE_H
#define __MPIMATRIXFILE_H

// ��������� �����, ����������� �������
typedef struct {
	unsigned char fourCC[4]; // ������������� ������� ����� http://en.wikipedia.org/wiki/FourCC
	MPI_Datatype dataType; // ��� ������ � �������
	int height; // ���������� �����
	int width; // ���������� ��������
	MPI_Offset offset; // ������� ������ � �����
} mpiMatrixHeader;

#endif