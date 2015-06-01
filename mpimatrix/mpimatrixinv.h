/*
* Нахождение обратной матрицы
* Каждый процесс считывает свой диапазон колонок
* Приписывается дополнительная единичная матрица
* Исходная матрица приводится преобразованиями Гаусса-Жордано к каноническому виду
* К дополнительной матрице применяем преобразования как и к основной
* Данная реализация метода Гаусса-Жордана не полностью приводит матрицу к ступенчатой
* Требуется дополнительная перестановка строк согласно индексам первых ненулевых элементов в строке
* Для нахождения определителя требуется умножить найденное произведение первых нунулевых элементов 
* на знак чётности дополнительной перестановки строк
* Сохраняет обратную матрицу из этих колонок
* Используются функции MPI_File_read_ordered и MPI_File_write_ordered
*/
#include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include "mpimatrixfile.h"
#include "mpimatrixgaussjordan.h"
#include "minmax.h"

#ifndef __MPIMATRIXINV_H
#define __MPIMATRIXINV_H

template <typename T>
void mpi_matrix_inv(
	char *inputFileName, 
	char *outputFileName, 
	long *counter) // Счётчик количества операций
{
	int np;    /* Общее количество процессов */
	int mp;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &mp);

	MPI_File file;
	MPI_Status status;
	mpiMatrixHeader header;
	int count;

	MPI_File_open(MPI_COMM_WORLD,inputFileName,MPI_MODE_RDONLY|MPI_MODE_SEQUENTIAL,MPI_INFO_NULL, &file);

	if(mp==0) memset( &status, 0x00, sizeof(MPI_Status) );
	if(mp==0) MPI_File_read_shared(file, &header, sizeof(mpiMatrixHeader), MPI_BYTE, &status);
	if(mp==0) MPI_Get_count( &status, MPI_INT, &count );
	if(np>1) MPI_Bcast(&header, sizeof(mpiMatrixHeader), MPI_BYTE, 0, MPI_COMM_WORLD);

	assert(header.height==header.width);
	assert(header.offset==sizeof(mpiMatrixHeader));

	MPI_Datatype dataType = header.dataType;
	MPI_Offset offset = header.offset;

	int rank = header.height;
	int wrank = min(np, rank); // Количество реально используемых процессов

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
	if(wrank<np) MPI_File_close(&file);
	if(mp>=wrank) return;
	if(wrank<np) MPI_File_open(comm,inputFileName,MPI_MODE_RDONLY,MPI_INFO_NULL, &file);
	if(wrank<np) MPI_File_seek_shared(file,offset,MPI_SEEK_SET);

	int start = rank*mp/wrank;
	int end = rank*(mp+1)/wrank;
	int length=end-start;
	int bufferSize=length*rank;

	T *buffer1 = (T*)malloc(sizeof(T)*bufferSize);
	T *buffer2 = (T*)malloc(sizeof(T)*bufferSize);

	// Считываем построчно исходный файл
	// Каждый процесс считывает только свой диапазон колонок
	for(int i=0;i<rank;i++){
		memset( &status, 0x00, sizeof(MPI_Status) );
		MPI_File_read_ordered( file, &buffer1[i*length], length, dataType, &status);
		MPI_Get_count( &status, MPI_INT, &count );
		// assert(count==length);
	}

	MPI_File_close(&file);

	T nil = (T)0;
	T one = (T)1;
	for(int i=0;i<bufferSize;i++) buffer2[i]=nil;
	
	for(int i=0;i<length;i++){
		buffer2[(start+i)*length+i]=one;
	}

	T totalDiv;
	int *localIndex = (int *)malloc(sizeof(int)*rank);
	int *totalIndex = (int *)malloc(sizeof(int)*rank);

	/*
	* Приведение матрицы к каноническому виду преобразованиями строк методом Гаусса-Жордана
	* Каждый процесс обрабатывает свой диапазон колонок
	* К дополнительной матрице применяем преобразования как и к основной
	* Данная реализация метода Гаусса-Жордана не полностью приводит матрицу к ступенчатой
	* Требуется дополнительная перестановка строк согласно индексам первых ненулевых элементов в строке
	* Для нахождения определителя требуется умножить найденное произведение первых нунулевых элементов 
	* на знак чётности дополнительной перестановки строк
	*/
	mpi_matrix_gaussjordan(
	comm, // Коммуникатор группы используемых процессов
	buffer1, // Локальный буфер столбцов основной матрицы
	buffer2, // Локальный буфер столбцов дополнительной матрицы
	&totalDiv, // Произведение первых нунулевых элементов приведённой основной матрицы
	localIndex, // Преаллокированный буфер для рассчёта индексов ненулевых элементов
	totalIndex, // Преаллокированный буфер для рассчёта индексов ненулевых элементов
	rank, rank, length, // Размеры основной матрицы и локальный размер дополнительной матрицы
	dataType, // Тип для обмена данными
	mp, wrank, // Номер процесса и количество используемых процессов
	counter); // Счётчик количества операций

	/*
	* Нахождение индексов первых ненулевых элементов в строках матрицы
	* Каждый процесс обрабатывает свой диапазон колонок
	* Используется MPI_Allreduce
	*/
	mpi_matrix_indexOfFirstNotZero(
	comm, // Коммуникатор группы используемых процессов
	buffer1, // Локальный буфер столбцов основной матрицы
	localIndex, // Преаллокированный буфер для рассчёта индексов ненулевых элементов
	totalIndex, // Преаллокированный буфер для рассчёта индексов ненулевых элементов
	rank, rank, // Размеры матрицы
	dataType, // Тип для обмена данными
	mp, wrank, // Номер процесса и количество используемых процессов
	counter); // Счётчик количества операций

	// Вычисление требуемой перестановки строк
	int *ii = (int *)malloc(sizeof(int)*rank);
	for(int i=0;i<rank;i++){
		ii[totalIndex[i]]=i;
	}

	if(mp==0) MPI_File_delete(outputFileName, MPI_INFO_NULL);
	MPI_File_open(comm,outputFileName,MPI_MODE_WRONLY|MPI_MODE_SEQUENTIAL|MPI_MODE_CREATE,MPI_INFO_NULL, &file);

	if(mp==0) memset( &status, 0x00, sizeof(MPI_Status) );
	if(mp==0) MPI_File_write_shared(file, &header, sizeof(mpiMatrixHeader), MPI_BYTE, &status);

	// Сохраняем построчно
	// Каждый процесс сохраняет только свой диапазон колонок
	// Порядок строк при сохранении определяется индексами первых ненулевых элементов
	for(int i=0;i<rank;i++){
		memset( &status, 0x00, sizeof(MPI_Status) );
		MPI_File_write_ordered(file, &buffer2[ii[i]*length], length, dataType, &status);
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
