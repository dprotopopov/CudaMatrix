#include "matrix.h"

int strempty(const char *p)
{
	if (!p)
		return (1);
	for (; *p; p++)
		if (!isspace(*p))
			return (0);
	return (1);
}

char *mystrtok(char **m, char *s, char c)
{
	char *p = s ? s : *m;
	if (!*p)
		return 0;
	*m = strchr(p, c);
	if (*m)
		*(*m)++ = 0;
	else
		*m = p + strlen(p);
	return p;
}

template<class T> void matrix_read(char *fileName, MATRIX<T> ** matrix) {
	char buffer[4096];
	char *tok;
	char *p;
	int i, j;

	FILE *fs = fopen(fileName, "r");

	if (fs == NULL) {
		fprintf(stderr, "File open error (%s)\n", fileName); fflush(stderr);
		exit(-1);
	}

	int height = 0;
	int width = 0;

	/* Заполняем массив числами из файла */
	/* Операция выполняетя в два прохода по файлу */
	/* На первом проходе определяется ранг матрицы */
	/* На втором проходе считываются данные */
	for (i = 0; (tok = fgets(buffer, sizeof(buffer), fs)) != NULL; i++)
	{
		j = 0;
		for (tok = mystrtok(&p, tok, ';'); tok != NULL; tok = mystrtok(&p, NULL, ';'))
		{
			j++;
		}
		width = max(width, j);
		height++;
	}

	*matrix = (MATRIX<T> *)malloc(sizeof(MATRIX<T>)+height*width * sizeof(T));
	(*matrix)->width = width;
	(*matrix)->height = height;

	fseek(fs, 0, SEEK_SET);

	for (int i = 0; (tok = fgets(buffer, sizeof(buffer), fs)) != NULL; i++)
	{
		int j = 0;
		for (tok = mystrtok(&p, tok, ';'); tok != NULL; tok = mystrtok(&p, NULL, ';'))
		{
			(*matrix)->values[IDX(i,j++,width)] = strempty(tok) ? (T)0 : (T)atof(tok);
		}
		for (; j < width; j++) (*matrix)->values[IDX(i,j,width)] = (T)0;
	}
	for (j = 0; j < (height - i)*(width); j++) (*matrix)->values[IDX(i,j,width)] = (T)0;

	fclose(fs);
}

template<class T> void matrix_write(char *fileName, MATRIX<T> * matrix) {
	int height = matrix->height;
	int width = matrix->width;
	int i, j;
	FILE *fs = fopen(fileName, "w");
	if (fs == NULL) {
		fprintf(stderr, "File open error (%s)\n", fileName); fflush(stderr);
		exit(-1);
	}
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			fprintf(fs, "%f%s", (double)matrix->values[IDX(i,j,width)], ((j == width - 1) ? "\n" : ";"));
		}
	}
	fclose(fs);
}

