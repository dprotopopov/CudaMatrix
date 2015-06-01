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
* Используются функции fread fwrite
*/
#include <mpi.h>
#include <stdio.h>
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
	FILE *log, long *counter) // Счётчик количества операций
{
	int np;    /* Общее количество процессов */
	int mp;    /* Номер текущего процесса */

	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &mp);

	FILE *file=NULL;
	mpiMatrixHeader header;

	if(mp==0) file = fopen(inputFileName,"rb");
	if(mp==0) fread(&header,1,sizeof(mpiMatrixHeader),file);
	if(np>1) MPI_Bcast(&header,sizeof(mpiMatrixHeader),MPI_BYTE,0,MPI_COMM_WORLD);

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

	if(mp>=wrank) return;

	int start = rank*mp/wrank;
	int end = rank*(mp+1)/wrank;
	int length=end-start;
	int bufferSize=rank;
	int bufferSize1=length*rank;
	int bufferSize2=length*rank;

	T *buffer = (T*)malloc(sizeof(T)*bufferSize+1);
	T *buffer1 = (T*)malloc(sizeof(T)*bufferSize1+1);
	T *buffer2 = (T*)malloc(sizeof(T)*bufferSize2+1);

	// Считываем построчно исходный файл
	// Каждый процесс считывает только свой диапазон колонок
	for(int i=0;i<rank;i++){
		if(mp==0) fread(buffer,sizeof(T),rank,file);
		for(int j=0;j<wrank;j++) {
			int start = rank*j/wrank;
			int end = rank*(j+1)/wrank;
			int count=end-start;
			if(j==0 && mp==0) memcpy(&buffer1[i*length],buffer,length*sizeof(T));
			else if(mp==0) MPI_Send(&buffer[start],count,dataType,j,0,comm);
			else if(mp==j) MPI_Recv(&buffer1[i*length],length,dataType,0,0,comm,MPI_STATUS_IGNORE);
		}
	}

	if(mp==0) fclose(file);

	T nil = (T)0;
	T one = (T)1;
	for(int i=0;i<bufferSize2;i++) buffer2[i]=nil;
	
	for(int i=0;i<length;i++){
		buffer2[(start+i)*length+i]=one;
	}

	T totalDiv;
	int *localIndex = (int *)malloc(sizeof(int)*rank);
	int *totalIndex = (int *)malloc(sizeof(int)*rank);

	// Начало записи лога
	fprintf(log,"process %d of %d\n", mp, np);
	fprintf(log,"function %s\n", __FUNCTION__); // http://stackoverflow.com/questions/679021/how-to-find-the-name-of-the-current-function-at-runtime

	fprintf(log,"operand:\n"); 
	for(int i=0;i<rank;i++) {
		for(int j=0;j<length;j++) fprintf(log,"%le\t", (double)buffer1[i*length+j]);
		fprintf(log,"\n");
	}

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
	log, counter); // Счётчик количества операций

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
	log, counter); // Счётчик количества операций

	// Вычисление требуемой перестановки строк
	int *ii = (int *)malloc(sizeof(int)*rank);
	for(int i=0;i<rank;i++){
		ii[totalIndex[i]]=i;
	}

	fprintf(log,"result:\n"); 
	for(int i=0;i<rank;i++) {
		for(int j=0;j<length;j++) fprintf(log,"%le\t", (double)buffer2[ii[i]*length+j]);
		fprintf(log,"\n");
	}

	fflush(log);
	// Завершение записи лога

	if(mp==0) file = fopen(outputFileName,"wb");
	if(mp==0) fwrite(&header,1,sizeof(mpiMatrixHeader),file);

	// Сохраняем построчно
	// Каждый процесс сохраняет только свой диапазон колонок
	// Порядок строк при сохранении определяется индексами первых ненулевых элементов
	for(int i=0;i<rank;i++){
		for(int j=0;j<wrank;j++) {
			int start = rank*j/wrank;
			int end = rank*(j+1)/wrank;
			int count=end-start;
			if(j==0 && mp==0) memcpy(buffer,&buffer2[ii[i]*length],length*sizeof(T));
			else if(mp==0) MPI_Recv(&buffer[start],count,dataType,j,0,comm,MPI_STATUS_IGNORE);
			else if(mp==j) MPI_Send(&buffer2[ii[i]*length],length,dataType,0,0,comm);
		}
		if(mp==0) fwrite(buffer,sizeof(T),rank,file);
	}

	if(mp==0) fclose(file);

	free(ii);
	free(localIndex);
	free(totalIndex);
	free(buffer);
	free(buffer1);
	free(buffer2);
}

#endif
