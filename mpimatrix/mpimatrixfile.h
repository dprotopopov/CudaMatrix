#include <mpi.h>

#ifndef __MPIMATRIXFILE_H
#define __MPIMATRIXFILE_H

// Заголовок файла, содержащего матрицу
typedef struct {
	unsigned char fourCC[4]; // Идентификатор формата файла http://en.wikipedia.org/wiki/FourCC
	MPI_Datatype dataType; // Тип данных в массиве
	int height; // Количество строк
	int width; // Количество столбцов
	MPI_Offset offset; // Позиция данных в файле
} mpiMatrixHeader;

#endif