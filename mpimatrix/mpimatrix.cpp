#define _CRT_SECURE_NO_WARNINGS

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "mpimatrixfile.h"
#include "mpimatrixmov.h"
#include "mpimatrixadd.h"
#include "mpimatrixsub.h"
#include "mpimatrixmul.h"
#include "mpimatrixrot.h"
#include "mpimatrixinv.h"
#include "mpimatrixdet.h"
#include "mpimatrixmtv.h"
#include "mpimatrixone.h"
#include "mpimatrixnil.h"
#include "mycom.h"
#include "mynet.h"
#include "myio.h"

static char pname[MPI_MAX_PROCESSOR_NAME];
static char vname[48] = "mpimatrix";
static char sname[48];
static FILE *Fi = NULL;
static FILE *Fo = NULL;

int main(int argc, char** argv){

	int np;     /* Общее количество процессов */
	int mp;    /* Номер текущего процесса */
	int nl, ier;
	double tick, time;

	MyNetInit(&argc,&argv,&np,&mp,&nl,pname,&tick);

	sprintf(sname,"%s.p%02d",vname,mp);
	ier = fopen_m(&Fo,sname,"wt");
	if (ier!=0) mpierr("Protocol file not opened",1);

	char *opCode = (argc>1)?argv[1]:NULL;
	char *fileName1 = (argc>2)?argv[2]:NULL;
	char *fileName2 = (argc>3)?argv[3]:NULL;
	char *fileName3 = (argc>4)?argv[4]:NULL;
	char *fileName4 = (argc>5)?argv[5]:NULL;
	char *rank = (argc>2)?argv[2]:NULL;
	char *a = (argc>2)?argv[2]:NULL;
	char *b = (argc>3)?argv[3]:NULL;
	char *height = (argc>2)?argv[2]:NULL;
	char *width = (argc>3)?argv[3]:NULL;

	long counter = 0; // Количество выполненых элементарных операций процессом

	time = MPI_Wtime(); // Начало отсчёта времени выполнения операции

	if(opCode!=NULL && strcmp(opCode,"add")==0 && fileName1!=NULL && fileName2!=NULL && fileName3!=NULL){
		mpi_matrix_add<double>(fileName1,fileName2,fileName3,&counter);
	}
	else if(opCode!=NULL && strcmp(opCode,"sub")==0 && fileName1!=NULL && fileName2!=NULL && fileName3!=NULL){
		mpi_matrix_sub<double>(fileName1,fileName2,fileName3,&counter);
	}
	else if(opCode!=NULL && strcmp(opCode,"mul")==0 && fileName1!=NULL && fileName2!=NULL && fileName3!=NULL){
		mpi_matrix_mul<double>(fileName1,fileName2,fileName3,&counter);
	}
	else if(opCode!=NULL && strcmp(opCode,"mtv")==0 && fileName1!=NULL && fileName2!=NULL && fileName3!=NULL){
		mpi_matrix_mtv<double>(fileName1,fileName2,fileName3,&counter);
	}
	else if(opCode!=NULL && strcmp(opCode,"inv")==0 && fileName1!=NULL && fileName2!=NULL){
		mpi_matrix_inv<double>(fileName1,fileName2,&counter);
	}
	else if(opCode!=NULL && strcmp(opCode,"mov")==0 && fileName1!=NULL && fileName2!=NULL){
		mpi_matrix_mov<double>(fileName1,fileName2,&counter);
	}
	else if(opCode!=NULL && strcmp(opCode,"rot")==0 && fileName1!=NULL && fileName2!=NULL){
		mpi_matrix_rot<double>(fileName1,fileName2,&counter);
	}
	else if(opCode!=NULL && strcmp(opCode,"det")==0 && fileName1!=NULL && fileName2!=NULL){
		mpi_matrix_det<double>(fileName1,fileName2,&counter);
	}
	else if(opCode!=NULL && strcmp(opCode,"one")==0 && rank!=NULL && fileName2!=NULL){
		mpi_matrix_one<double>(atoi(rank),fileName2,&counter);
	}
	else if(opCode!=NULL && strcmp(opCode,"nil")==0 && height!=NULL && width!=NULL && fileName3!=NULL){
		mpi_matrix_nil<double>(atoi(height),atoi(width),fileName3,&counter);
	}
	else if (mp == 0) {
		printf("Usage :\t%s add inputfilename1 inputfilename2 outputfilename \n", argv[0]);
		printf("Usage :\t%s sub inputfilename1 inputfilename2 outputfilename \n", argv[0]);
		printf("Usage :\t%s mul inputfilename1 inputfilename2 outputfilename \n", argv[0]);
		printf("Usage :\t%s mtv inputfilename1(1x1) inputfilename2 outputfilename \n", argv[0]);
		printf("Usage :\t%s inv inputfilename outputfilename \n", argv[0]);
		printf("Usage :\t%s mov inputfilename outputfilename \n", argv[0]);
		printf("Usage :\t%s rot inputfilename outputfilename \n", argv[0]);
		printf("Usage :\t%s det inputfilename outputfilename(1x1) \n", argv[0]);
		printf("Usage :\t%s one rank outputfilename \n", argv[0]);
		printf("Usage :\t%s nil height width outputfilename \n", argv[0]);
		MPI_Finalize();
		exit(-1);
	}
	else {
		MPI_Finalize();
		exit(-1);
	}

	time = MPI_Wtime() - time; // Окончание отсчёта времени выполнения операции

	fprintf(Fo,"%d of %d process execute %ld operations %le seconds\n", mp, np, counter, time);fflush(Fo);

	long totalCounter = 0; // Количество выполненых элементарных операций программой
	MPI_Allreduce(&counter,&totalCounter,1,MPI_LONG,MPI_SUM,MPI_COMM_WORLD);

	if (mp == 0) {
		// Сохранение записи о количестве выполненных элементарных операций и времени выполнения операции
		printf("%d process execute %ld operations %le seconds\n", np, totalCounter, time);
	}

	ier = fclose_m(&Fo);

	MPI_Finalize();
	exit(0);
}
