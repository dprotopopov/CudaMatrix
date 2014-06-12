/* A Bison parser, made by GNU Bison 3.0.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2013 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_YY_MATRIX_TAB_H_INCLUDED
# define YY_YY_MATRIX_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    SHOW_KEYWORD = 258,
    INFO_KEYWORD = 259,
    HELP_KEYWORD = 260,
    READ_KEYWORD = 261,
    WRITE_KEYWORD = 262,
    ECHO_KEYWORD = 263,
    LET_KEYWORD = 264,
    SET_KEYWORD = 265,
    USE_KEYWORD = 266,
    ALLOC_KEYWORD = 267,
    FREE_KEYWORD = 268,
    SRC_KEYWORD = 269,
    DEST_KEYWORD = 270,
    CACHE_KEYWORD = 271,
    INTEGER = 272,
    DOUBLE = 273,
    VARIABLE = 274,
    STRING = 275,
    TEXTURE_KEYWORD = 276,
    CONSTANT_KEYWORD = 277,
    GLOBAL_KEYWORD = 278,
    SHARED_KEYWORD = 279,
    LOCAL_KEYWORD = 280,
    NONE_KEYWORD = 281,
    BLOCKS_KEYWORD = 282,
    THREADS_KEYWORD = 283,
    TOLERANCE_KEYWORD = 284,
    GAUSSJORDAN_KEYWORD = 285,
    ROT_KEYWORD = 286,
    INV_KEYWORD = 287
  };
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef int YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;

int yyparse (void);

#endif /* !YY_YY_MATRIX_TAB_H_INCLUDED  */