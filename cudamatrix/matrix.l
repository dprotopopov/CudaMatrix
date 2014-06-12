%{
#define _CRT_SECURE_NO_WARNINGS

#include <string.h>
#include "matrix.tab.h"
#include "matrix.h"

void yyerror(char *);

%}
%%
\"[^\"]*\" {
	int len = strlen(yytext)-2;
	char * buffer = (char *)malloc(len+1);
	strncpy(buffer, &yytext[1], len);
	buffer[len] = '\0';
	string_stack[string_stack_size] = buffer;
	yylval = string_stack_size++;
	return STRING;
}
show { return SHOW_KEYWORD; }
info { return INFO_KEYWORD; }
help { return HELP_KEYWORD; }
echo { return ECHO_KEYWORD; }
let { return LET_KEYWORD; }
set { return SET_KEYWORD; }
use { return USE_KEYWORD; }
alloc { return ALLOC_KEYWORD; }
free { return FREE_KEYWORD; }
read { return READ_KEYWORD; }
write { return WRITE_KEYWORD; }
src { return SRC_KEYWORD; }
source { return SRC_KEYWORD; }
dest { return DEST_KEYWORD; }
cache { return CACHE_KEYWORD; }
blocks { return BLOCKS_KEYWORD; }
threads { return THREADS_KEYWORD; }
rot { return ROT_KEYWORD; }
inv { return INV_KEYWORD; }
gaussjordan { return GAUSSJORDAN_KEYWORD; }
texture { return TEXTURE_KEYWORD; }
constant { return CONSTANT_KEYWORD; }
global { return GLOBAL_KEYWORD; }
shared { return SHARED_KEYWORD; }
local { return LOCAL_KEYWORD; }
none { return NONE_KEYWORD; }
tolerance { return TOLERANCE_KEYWORD; }
[a-z] {
	yylval = *yytext - 'a';
	return VARIABLE;
}
[0-9]+\.[0-9]* {
	double_stack[double_stack_size++] = atof(yytext);
	yylval = double_stack_size;
	return DOUBLE;
}
[0-9]+ {
	yylval = atoi(yytext);
	return INTEGER;
}
[-+()=*\n] { return *yytext; }
[ \t] ; /* skip whitespace */
. yyerror("Unknown character");
%%
int yywrap(void) {
	return 1;
}