/* A Bison parser, made by GNU Bison 3.0.  */

/* Bison implementation for Yacc-like parsers in C

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

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "3.0"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1




/* Copy the first part of user declarations.  */
#line 1 "matrix.y" /* yacc.c:339  */

#include "matrix.h"

void yyerror(char *);
int yylex(void);
MATRIX<double> * sym[26];
MEMORY source_memory = TEXTURE;
MEMORY dest_memory = GLOBAL;
MEMORY cache_memory = SHARED;
dim3 blocks = dim3(1,1,1);
dim3 threads = dim3(1,1,1);
double tolerance = 0.0;

#line 80 "matrix.tab.cpp" /* yacc.c:339  */

# ifndef YY_NULL
#  if defined __cplusplus && 201103L <= __cplusplus
#   define YY_NULL nullptr
#  else
#   define YY_NULL 0
#  endif
# endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* In a future release of Bison, this section will be replaced
   by #include "matrix.tab.h".  */
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

/* Copy the second part of user declarations.  */

#line 164 "matrix.tab.cpp" /* yacc.c:358  */

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#else
typedef signed char yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if defined YYENABLE_NLS && YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(Msgid) dgettext ("bison-runtime", Msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(Msgid) Msgid
# endif
#endif

#ifndef __attribute__
/* This feature is available in gcc versions 2.5 and later.  */
# if (! defined __GNUC__ || __GNUC__ < 2 \
      || (__GNUC__ == 2 && __GNUC_MINOR__ < 5))
#  define __attribute__(Spec) /* empty */
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(E) ((void) (E))
#else
# define YYUSE(E) /* empty */
#endif

#if defined __GNUC__ && 407 <= __GNUC__ * 100 + __GNUC_MINOR__
/* Suppress an incorrect diagnostic about yylval being uninitialized.  */
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN \
    _Pragma ("GCC diagnostic push") \
    _Pragma ("GCC diagnostic ignored \"-Wuninitialized\"")\
    _Pragma ("GCC diagnostic ignored \"-Wmaybe-uninitialized\"")
# define YY_IGNORE_MAYBE_UNINITIALIZED_END \
    _Pragma ("GCC diagnostic pop")
#else
# define YY_INITIAL_VALUE(Value) Value
#endif
#ifndef YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
# define YY_IGNORE_MAYBE_UNINITIALIZED_END
#endif
#ifndef YY_INITIAL_VALUE
# define YY_INITIAL_VALUE(Value) /* Nothing. */
#endif


#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined EXIT_SUCCESS
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
      /* Use EXIT_SUCCESS as a witness for stdlib.h.  */
#     ifndef EXIT_SUCCESS
#      define EXIT_SUCCESS 0
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's 'empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (0)
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined EXIT_SUCCESS \
       && ! ((defined YYMALLOC || defined malloc) \
             && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef EXIT_SUCCESS
#    define EXIT_SUCCESS 0
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined EXIT_SUCCESS
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined EXIT_SUCCESS
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
         || (defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE)) \
      + YYSTACK_GAP_MAXIMUM)

# define YYCOPY_NEEDED 1

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)                           \
    do                                                                  \
      {                                                                 \
        YYSIZE_T yynewbytes;                                            \
        YYCOPY (&yyptr->Stack_alloc, Stack, yysize);                    \
        Stack = &yyptr->Stack_alloc;                                    \
        yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
        yyptr += yynewbytes / sizeof (*yyptr);                          \
      }                                                                 \
    while (0)

#endif

#if defined YYCOPY_NEEDED && YYCOPY_NEEDED
/* Copy COUNT objects from SRC to DST.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(Dst, Src, Count) \
      __builtin_memcpy (Dst, Src, (Count) * sizeof (*(Src)))
#  else
#   define YYCOPY(Dst, Src, Count)              \
      do                                        \
        {                                       \
          YYSIZE_T yyi;                         \
          for (yyi = 0; yyi < (Count); yyi++)   \
            (Dst)[yyi] = (Src)[yyi];            \
        }                                       \
      while (0)
#  endif
# endif
#endif /* !YYCOPY_NEEDED */

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  2
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   63

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  38
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  3
/* YYNRULES -- Number of rules.  */
#define YYNRULES  30
/* YYNSTATES -- Number of states.  */
#define YYNSTATES  65

/* YYTRANSLATE[YYX] -- Symbol number corresponding to YYX as returned
   by yylex, with out-of-bounds checking.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   287

#define YYTRANSLATE(YYX)                                                \
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[TOKEN-NUM] -- Symbol number corresponding to TOKEN-NUM
   as returned by yylex, without out-of-bounds checking.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      36,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,    35,    33,     2,    34,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    37,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32
};

#if YYDEBUG
  /* YYRLINE[YYN] -- Source line where rule number YYN was defined.  */
static const yytype_uint8 yyrline[] =
{
       0,    26,    26,    27,    30,    31,    40,    49,    58,    67,
      76,    85,    94,   106,   118,   130,   131,   132,   133,   134,
     135,   136,   137,   138,   139,   140,   141,   142,   147,   148,
     239
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || 0
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "SHOW_KEYWORD", "INFO_KEYWORD",
  "HELP_KEYWORD", "READ_KEYWORD", "WRITE_KEYWORD", "ECHO_KEYWORD",
  "LET_KEYWORD", "SET_KEYWORD", "USE_KEYWORD", "ALLOC_KEYWORD",
  "FREE_KEYWORD", "SRC_KEYWORD", "DEST_KEYWORD", "CACHE_KEYWORD",
  "INTEGER", "DOUBLE", "VARIABLE", "STRING", "TEXTURE_KEYWORD",
  "CONSTANT_KEYWORD", "GLOBAL_KEYWORD", "SHARED_KEYWORD", "LOCAL_KEYWORD",
  "NONE_KEYWORD", "BLOCKS_KEYWORD", "THREADS_KEYWORD", "TOLERANCE_KEYWORD",
  "GAUSSJORDAN_KEYWORD", "ROT_KEYWORD", "INV_KEYWORD", "'+'", "'-'", "'*'",
  "'\\n'", "'='", "$accept", "program", "statement", YY_NULL
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[NUM] -- (External) token number corresponding to the
   (internal) symbol number NUM (which must be that of a token).  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,    43,    45,    42,    10,    61
};
# endif

#define YYPACT_NINF -27

#define yypact_value_is_default(Yystate) \
  (!!((Yystate) == (-27)))

#define YYTABLE_NINF -1

#define yytable_value_is_error(Yytable_value) \
  0

  /* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
     STATE-NUM.  */
static const yytype_int8 yypact[] =
{
     -27,    16,   -27,    13,   -27,    -9,    17,    18,    20,   -26,
     -10,    21,    22,     7,   -27,    24,    25,   -27,     5,    29,
      30,    19,   -14,    26,     6,    31,   -27,   -27,   -27,   -27,
     -19,    33,    34,   -27,   -27,   -27,   -27,   -27,   -27,   -27,
     -27,    35,     0,     1,    36,    37,    38,    39,   -27,   -27,
     -27,    40,    41,    42,    45,   -27,   -27,   -27,   -27,   -27,
     -27,   -27,   -27,    44,   -27
};

  /* YYDEFACT[STATE-NUM] -- Default reduction number in state STATE-NUM.
     Performed when YYTABLE does not specify something else to do.  Zero
     means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       3,     0,     1,     0,    30,     0,     0,     0,     0,     0,
       0,     0,     0,     0,    29,     0,     0,     4,     0,     0,
       0,     0,     0,     0,     0,     0,    28,     2,    15,    16,
       0,     0,     0,    26,    17,    18,    19,    20,    23,    22,
      21,     0,     5,     0,     0,     0,     0,     0,    24,    25,
      27,     0,     0,     0,     0,    11,     8,     9,     6,     7,
      12,    13,    14,     0,    10
};

  /* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -27,   -27,   -27
};

  /* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,     1,    13
};

  /* YYTABLE[YYPACT[STATE-NUM]] -- What to do in state STATE-NUM.  If
     positive, shift that token.  If negative, reduce the rule whose
     number is the opposite.  If YYTABLE_NINF, syntax error.  */
static const yytype_uint8 yytable[] =
{
      42,    19,    20,    21,    22,    23,    24,    34,    35,    36,
      15,    43,    44,    45,    46,    47,     2,    14,    54,     3,
      55,     4,     5,     6,     7,     8,     9,    10,    11,    12,
      38,    39,    40,    51,    52,    53,    16,    33,    17,    18,
      25,    26,    30,    27,    28,    29,    31,    32,    41,    37,
      48,    49,    50,     0,     0,    56,    57,    58,    59,    60,
      61,    62,    63,    64
};

static const yytype_int8 yycheck[] =
{
      19,    27,    28,    29,    14,    15,    16,    21,    22,    23,
      19,    30,    31,    32,    33,    34,     0,     4,    17,     3,
      19,     5,     6,     7,     8,     9,    10,    11,    12,    13,
      24,    25,    26,    33,    34,    35,    19,    18,    20,    19,
      19,    19,    37,    36,    20,    20,    17,    17,    17,    23,
      17,    17,    17,    -1,    -1,    19,    19,    19,    19,    19,
      19,    19,    17,    19
};

  /* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
     symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,    39,     0,     3,     5,     6,     7,     8,     9,    10,
      11,    12,    13,    40,     4,    19,    19,    20,    19,    27,
      28,    29,    14,    15,    16,    19,    19,    36,    20,    20,
      37,    17,    17,    18,    21,    22,    23,    23,    24,    25,
      26,    17,    19,    30,    31,    32,    33,    34,    17,    17,
      17,    33,    34,    35,    17,    19,    19,    19,    19,    19,
      19,    19,    19,    17,    19
};

  /* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    38,    39,    39,    40,    40,    40,    40,    40,    40,
      40,    40,    40,    40,    40,    40,    40,    40,    40,    40,
      40,    40,    40,    40,    40,    40,    40,    40,    40,    40,
      40
};

  /* YYR2[YYN] -- Number of symbols on the right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     3,     0,     2,     4,     5,     5,     5,     5,
       7,     5,     6,     6,     6,     3,     3,     3,     3,     3,
       3,     3,     3,     3,     4,     4,     3,     4,     2,     2,
       1
};


#define yyerrok         (yyerrstatus = 0)
#define yyclearin       (yychar = YYEMPTY)
#define YYEMPTY         (-2)
#define YYEOF           0

#define YYACCEPT        goto yyacceptlab
#define YYABORT         goto yyabortlab
#define YYERROR         goto yyerrorlab


#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)                                  \
do                                                              \
  if (yychar == YYEMPTY)                                        \
    {                                                           \
      yychar = (Token);                                         \
      yylval = (Value);                                         \
      YYPOPSTACK (yylen);                                       \
      yystate = *yyssp;                                         \
      goto yybackup;                                            \
    }                                                           \
  else                                                          \
    {                                                           \
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;                                                  \
    }                                                           \
while (0)

/* Error token number */
#define YYTERROR        1
#define YYERRCODE       256



/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)                        \
do {                                            \
  if (yydebug)                                  \
    YYFPRINTF Args;                             \
} while (0)

/* This macro is provided for backward compatibility. */
#ifndef YY_LOCATION_PRINT
# define YY_LOCATION_PRINT(File, Loc) ((void) 0)
#endif


# define YY_SYMBOL_PRINT(Title, Type, Value, Location)                    \
do {                                                                      \
  if (yydebug)                                                            \
    {                                                                     \
      YYFPRINTF (stderr, "%s ", Title);                                   \
      yy_symbol_print (stderr,                                            \
                  Type, Value); \
      YYFPRINTF (stderr, "\n");                                           \
    }                                                                     \
} while (0)


/*----------------------------------------.
| Print this symbol's value on YYOUTPUT.  |
`----------------------------------------*/

static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  FILE *yyo = yyoutput;
  YYUSE (yyo);
  if (!yyvaluep)
    return;
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# endif
  YYUSE (yytype);
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep)
{
  YYFPRINTF (yyoutput, "%s %s (",
             yytype < YYNTOKENS ? "token" : "nterm", yytname[yytype]);

  yy_symbol_value_print (yyoutput, yytype, yyvaluep);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)                            \
do {                                                            \
  if (yydebug)                                                  \
    yy_stack_print ((Bottom), (Top));                           \
} while (0)


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

static void
yy_reduce_print (yytype_int16 *yyssp, YYSTYPE *yyvsp, int yyrule)
{
  unsigned long int yylno = yyrline[yyrule];
  int yynrhs = yyr2[yyrule];
  int yyi;
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
             yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr,
                       yystos[yyssp[yyi + 1 - yynrhs]],
                       &(yyvsp[(yyi + 1) - (yynrhs)])
                                              );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)          \
do {                                    \
  if (yydebug)                          \
    yy_reduce_print (yyssp, yyvsp, Rule); \
} while (0)

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif


#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
static YYSIZE_T
yystrlen (const char *yystr)
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
static char *
yystpcpy (char *yydest, const char *yysrc)
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into *YYMSG, which is of size *YYMSG_ALLOC, an error message
   about the unexpected token YYTOKEN for the state stack whose top is
   YYSSP.

   Return 0 if *YYMSG was successfully written.  Return 1 if *YYMSG is
   not large enough to hold the message.  In that case, also set
   *YYMSG_ALLOC to the required number of bytes.  Return 2 if the
   required number of bytes is too large to store.  */
static int
yysyntax_error (YYSIZE_T *yymsg_alloc, char **yymsg,
                yytype_int16 *yyssp, int yytoken)
{
  YYSIZE_T yysize0 = yytnamerr (YY_NULL, yytname[yytoken]);
  YYSIZE_T yysize = yysize0;
  enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
  /* Internationalized format string. */
  const char *yyformat = YY_NULL;
  /* Arguments of yyformat. */
  char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
  /* Number of reported tokens (one for the "unexpected", one per
     "expected"). */
  int yycount = 0;

  /* There are many possibilities here to consider:
     - If this state is a consistent state with a default action, then
       the only way this function was invoked is if the default action
       is an error action.  In that case, don't check for expected
       tokens because there are none.
     - The only way there can be no lookahead present (in yychar) is if
       this state is a consistent state with a default action.  Thus,
       detecting the absence of a lookahead is sufficient to determine
       that there is no unexpected or expected token to report.  In that
       case, just report a simple "syntax error".
     - Don't assume there isn't a lookahead just because this state is a
       consistent state with a default action.  There might have been a
       previous inconsistent state, consistent state with a non-default
       action, or user semantic action that manipulated yychar.
     - Of course, the expected token list depends on states to have
       correct lookahead information, and it depends on the parser not
       to perform extra reductions after fetching a lookahead from the
       scanner and before detecting a syntax error.  Thus, state merging
       (from LALR or IELR) and default reductions corrupt the expected
       token list.  However, the list is correct for canonical LR with
       one exception: it will still contain any token that will not be
       accepted due to an error action in a later state.
  */
  if (yytoken != YYEMPTY)
    {
      int yyn = yypact[*yyssp];
      yyarg[yycount++] = yytname[yytoken];
      if (!yypact_value_is_default (yyn))
        {
          /* Start YYX at -YYN if negative to avoid negative indexes in
             YYCHECK.  In other words, skip the first -YYN actions for
             this state because they are default actions.  */
          int yyxbegin = yyn < 0 ? -yyn : 0;
          /* Stay within bounds of both yycheck and yytname.  */
          int yychecklim = YYLAST - yyn + 1;
          int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
          int yyx;

          for (yyx = yyxbegin; yyx < yyxend; ++yyx)
            if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR
                && !yytable_value_is_error (yytable[yyx + yyn]))
              {
                if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
                  {
                    yycount = 1;
                    yysize = yysize0;
                    break;
                  }
                yyarg[yycount++] = yytname[yyx];
                {
                  YYSIZE_T yysize1 = yysize + yytnamerr (YY_NULL, yytname[yyx]);
                  if (! (yysize <= yysize1
                         && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
                    return 2;
                  yysize = yysize1;
                }
              }
        }
    }

  switch (yycount)
    {
# define YYCASE_(N, S)                      \
      case N:                               \
        yyformat = S;                       \
      break
      YYCASE_(0, YY_("syntax error"));
      YYCASE_(1, YY_("syntax error, unexpected %s"));
      YYCASE_(2, YY_("syntax error, unexpected %s, expecting %s"));
      YYCASE_(3, YY_("syntax error, unexpected %s, expecting %s or %s"));
      YYCASE_(4, YY_("syntax error, unexpected %s, expecting %s or %s or %s"));
      YYCASE_(5, YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s"));
# undef YYCASE_
    }

  {
    YYSIZE_T yysize1 = yysize + yystrlen (yyformat);
    if (! (yysize <= yysize1 && yysize1 <= YYSTACK_ALLOC_MAXIMUM))
      return 2;
    yysize = yysize1;
  }

  if (*yymsg_alloc < yysize)
    {
      *yymsg_alloc = 2 * yysize;
      if (! (yysize <= *yymsg_alloc
             && *yymsg_alloc <= YYSTACK_ALLOC_MAXIMUM))
        *yymsg_alloc = YYSTACK_ALLOC_MAXIMUM;
      return 1;
    }

  /* Avoid sprintf, as that infringes on the user's name space.
     Don't have undefined behavior even if the translation
     produced a string with the wrong number of "%s"s.  */
  {
    char *yyp = *yymsg;
    int yyi = 0;
    while ((*yyp = *yyformat) != '\0')
      if (*yyp == '%' && yyformat[1] == 's' && yyi < yycount)
        {
          yyp += yytnamerr (yyp, yyarg[yyi++]);
          yyformat += 2;
        }
      else
        {
          yyp++;
          yyformat++;
        }
  }
  return 0;
}
#endif /* YYERROR_VERBOSE */

/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep)
{
  YYUSE (yyvaluep);
  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  YYUSE (yytype);
  YY_IGNORE_MAYBE_UNINITIALIZED_END
}




/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;
/* Number of syntax errors so far.  */
int yynerrs;


/*----------.
| yyparse.  |
`----------*/

int
yyparse (void)
{
    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       'yyss': related to states.
       'yyvs': related to semantic values.

       Refer to the stacks through separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken = 0;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yyssp = yyss = yyssa;
  yyvsp = yyvs = yyvsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */
  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
        /* Give user a chance to reallocate the stack.  Use copies of
           these so that the &'s don't force the real ones into
           memory.  */
        YYSTYPE *yyvs1 = yyvs;
        yytype_int16 *yyss1 = yyss;

        /* Each stack pointer address is followed by the size of the
           data in use in that stack, in bytes.  This used to be a
           conditional around just the two extra args, but that might
           be undefined if yyoverflow is a macro.  */
        yyoverflow (YY_("memory exhausted"),
                    &yyss1, yysize * sizeof (*yyssp),
                    &yyvs1, yysize * sizeof (*yyvsp),
                    &yystacksize);

        yyss = yyss1;
        yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
        goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
        yystacksize = YYMAXDEPTH;

      {
        yytype_int16 *yyss1 = yyss;
        union yyalloc *yyptr =
          (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
        if (! yyptr)
          goto yyexhaustedlab;
        YYSTACK_RELOCATE (yyss_alloc, yyss);
        YYSTACK_RELOCATE (yyvs_alloc, yyvs);
#  undef YYSTACK_RELOCATE
        if (yyss1 != yyssa)
          YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
                  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
        YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yypact_value_is_default (yyn))
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = yylex ();
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yytable_value_is_error (yyn))
        goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END

  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     '$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];


  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 4:
#line 30 "matrix.y" /* yacc.c:1646  */
    { printf("%s\n",string_stack[(yyvsp[0])]); }
#line 1262 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 5:
#line 31 "matrix.y" /* yacc.c:1646  */
    { 
			if ((yyvsp[-1])==(yyvsp[0])) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[(yyvsp[0])]->height;
			int width = sym[(yyvsp[0])]->width;
			sym[(yyvsp[-2])] = (MATRIX<double> *)malloc(sizeof(MATRIX<double>)+(width * height * sizeof(double)));
			sym[(yyvsp[-2])]->height = height;
			sym[(yyvsp[-2])]->width = width;
			memcpy(sym[(yyvsp[0])]->values,sym[(yyvsp[-2])]->values,height*width*sizeof(double)); 
		}
#line 1276 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 6:
#line 40 "matrix.y" /* yacc.c:1646  */
    { 
			if ((yyvsp[-3])==(yyvsp[0])) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[(yyvsp[0])]->height;
			int width = sym[(yyvsp[0])]->width;
			sym[(yyvsp[-3])] = (MATRIX<double> *)malloc(sizeof(MATRIX<double>)+(width * height * sizeof(double)));
			sym[(yyvsp[-3])]->height = height;
			sym[(yyvsp[-3])]->width = width;
			__host__matrix_plus<double>(blocks,threads,sym[(yyvsp[0])],sym[(yyvsp[-3])],source_memory,dest_memory,cache_memory); 
		}
#line 1290 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 7:
#line 49 "matrix.y" /* yacc.c:1646  */
    { 
			if ((yyvsp[-3])==(yyvsp[0])) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[(yyvsp[0])]->height;
			int width = sym[(yyvsp[0])]->width;
			sym[(yyvsp[-3])] = (MATRIX<double> *)malloc(sizeof(MATRIX<double>)+(width * height * sizeof(double)));
			sym[(yyvsp[-3])]->height = height;
			sym[(yyvsp[-3])]->width = width;
			__host__matrix_minus<double>(blocks,threads,sym[(yyvsp[0])],sym[(yyvsp[-3])],source_memory,dest_memory,cache_memory); 
		}
#line 1304 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 8:
#line 58 "matrix.y" /* yacc.c:1646  */
    { 
			if ((yyvsp[-3])==(yyvsp[0])) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[(yyvsp[0])]->height;
			int width = sym[(yyvsp[0])]->width;
			sym[(yyvsp[-3])] = (MATRIX<double> *)malloc(sizeof(MATRIX<double>)+(width * height * sizeof(double)));
			sym[(yyvsp[-3])]->height = height;
			sym[(yyvsp[-3])]->width = width;
			__host__matrix_rot<double>(blocks,threads,sym[(yyvsp[0])],sym[(yyvsp[-3])],source_memory,dest_memory,cache_memory); 
		}
#line 1318 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 9:
#line 67 "matrix.y" /* yacc.c:1646  */
    { 
			if ((yyvsp[-3])==(yyvsp[0])) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[(yyvsp[0])]->height;
			int width = sym[(yyvsp[0])]->width;
			sym[(yyvsp[-3])] = (MATRIX<double> *)malloc(sizeof(MATRIX<double>)+(width * height * sizeof(double)));
			sym[(yyvsp[-3])]->height = height;
			sym[(yyvsp[-3])]->width = width;
			__host__matrix_inv<double>(blocks,threads,sym[(yyvsp[0])],sym[(yyvsp[-3])],tolerance,source_memory,dest_memory,cache_memory); 
		}
#line 1332 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 10:
#line 76 "matrix.y" /* yacc.c:1646  */
    { 
			if ((yyvsp[-5])==(yyvsp[0])) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[(yyvsp[0])]->height;
			int width = sym[(yyvsp[0])]->width;
			sym[(yyvsp[-5])] = (MATRIX<double> *)malloc(sizeof(MATRIX<double>)+(width * height * sizeof(double)));
			sym[(yyvsp[-5])]->height = height;
			sym[(yyvsp[-5])]->width = width;
			__host__matrix_gaussjordanstep<double>(blocks,threads,sym[(yyvsp[0])],sym[(yyvsp[-6])],(yyvsp[-2]),(yyvsp[-1]),source_memory,dest_memory,cache_memory); 
		}
#line 1346 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 11:
#line 85 "matrix.y" /* yacc.c:1646  */
    { 
			if ((yyvsp[-3])==(yyvsp[0])) { yyerror("L-value must be different from R-value"); exit(-1); }
			int height = sym[(yyvsp[0])]->height;
			int width = sym[(yyvsp[0])]->width;
			sym[(yyvsp[-3])] = (MATRIX<double> *)malloc(sizeof(MATRIX<double>)+(width * height * sizeof(double)));
			sym[(yyvsp[-3])]->height = height;
			sym[(yyvsp[-3])]->width = width;
			__host__matrix_gaussjordan<double>(blocks,threads,sym[(yyvsp[0])],sym[(yyvsp[-3])],tolerance,source_memory,dest_memory,cache_memory); 
		}
#line 1360 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 12:
#line 94 "matrix.y" /* yacc.c:1646  */
    { 
			if ((yyvsp[-4])==(yyvsp[-2])) { yyerror("L-value must be different from R-value"); exit(-1); }
			if ((yyvsp[-4])==(yyvsp[0])) { yyerror("L-value must be different from R-value"); exit(-1); }
			if (sym[(yyvsp[-2])]->height!=sym[(yyvsp[0])]->height) { yyerror("First argument height must be equal second argument height"); exit(-1); }
			if (sym[(yyvsp[-2])]->width!=sym[(yyvsp[0])]->width) { yyerror("First argument width must be equal second argument width"); exit(-1); }
			int height = sym[(yyvsp[-2])]->height;
			int width = sym[(yyvsp[0])]->width;
			sym[(yyvsp[-4])] = (MATRIX<double> *)malloc(sizeof(MATRIX<double>)+(width * height * sizeof(double)));
			sym[(yyvsp[-4])]->height = height;
			sym[(yyvsp[-4])]->width = width;
			__host__matrix_add<double>(blocks,threads,sym[(yyvsp[-2])],sym[(yyvsp[0])],sym[(yyvsp[-4])],source_memory,dest_memory,cache_memory); 
		}
#line 1377 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 13:
#line 106 "matrix.y" /* yacc.c:1646  */
    { 
			if ((yyvsp[-4])==(yyvsp[-2])) { yyerror("L-value must be different from R-value"); exit(-1); }
			if ((yyvsp[-4])==(yyvsp[0])) { yyerror("L-value must be different from R-value"); exit(-1); }
			if (sym[(yyvsp[-2])]->height!=sym[(yyvsp[0])]->height) { yyerror("First argument height must be equal second argument height"); exit(-1); }
			if (sym[(yyvsp[-2])]->width!=sym[(yyvsp[0])]->width) { yyerror("First argument width must be equal second argument width"); exit(-1); }
			int height = sym[(yyvsp[-2])]->height;
			int width = sym[(yyvsp[0])]->width;
			sym[(yyvsp[-4])] = (MATRIX<double> *)malloc(sizeof(MATRIX<double>)+(width * height * sizeof(double)));
			sym[(yyvsp[-4])]->height = height;
			sym[(yyvsp[-4])]->width = width;
			__host__matrix_sub<double>(blocks,threads,sym[(yyvsp[-2])],sym[(yyvsp[0])],sym[(yyvsp[-4])],source_memory,dest_memory,cache_memory); 
		}
#line 1394 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 14:
#line 118 "matrix.y" /* yacc.c:1646  */
    { 
			if ((yyvsp[-4])==(yyvsp[-2])) { yyerror("L-value must be different from R-value"); exit(-1); }
			if ((yyvsp[-4])==(yyvsp[0])) { yyerror("L-value must be different from R-value"); exit(-1); }
			if (sym[(yyvsp[-2])]->width!=sym[(yyvsp[0])]->height) { yyerror("First argument width must be equal second argument height"); exit(-1); }
			int height = sym[(yyvsp[-2])]->height;
			int width_height = sym[(yyvsp[-2])]->width;
			int width = sym[(yyvsp[0])]->width;
			sym[(yyvsp[-4])] = (MATRIX<double> *)malloc(sizeof(MATRIX<double>)+(width * height * sizeof(double)));
			sym[(yyvsp[-4])]->height = height;
			sym[(yyvsp[-4])]->width = width;
			__host__matrix_mul<double>(blocks,threads,sym[(yyvsp[-2])],sym[(yyvsp[0])],sym[(yyvsp[-4])],source_memory,dest_memory,cache_memory); 
		}
#line 1411 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 15:
#line 130 "matrix.y" /* yacc.c:1646  */
    { matrix_read<double>(string_stack[(yyvsp[0])], &sym[(yyvsp[-1])]); }
#line 1417 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 16:
#line 131 "matrix.y" /* yacc.c:1646  */
    { matrix_write<double>(string_stack[(yyvsp[0])], sym[(yyvsp[-1])]); }
#line 1423 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 17:
#line 132 "matrix.y" /* yacc.c:1646  */
    { source_memory = TEXTURE; }
#line 1429 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 18:
#line 133 "matrix.y" /* yacc.c:1646  */
    { source_memory = CONSTANT; }
#line 1435 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 19:
#line 134 "matrix.y" /* yacc.c:1646  */
    { source_memory = GLOBAL; }
#line 1441 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 20:
#line 135 "matrix.y" /* yacc.c:1646  */
    { dest_memory = GLOBAL; }
#line 1447 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 21:
#line 136 "matrix.y" /* yacc.c:1646  */
    { cache_memory = NONE; }
#line 1453 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 22:
#line 137 "matrix.y" /* yacc.c:1646  */
    { cache_memory = LOCAL; }
#line 1459 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 23:
#line 138 "matrix.y" /* yacc.c:1646  */
    { cache_memory = SHARED; }
#line 1465 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 24:
#line 139 "matrix.y" /* yacc.c:1646  */
    { blocks = dim3((yyvsp[-1]),(yyvsp[0]),1); }
#line 1471 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 25:
#line 140 "matrix.y" /* yacc.c:1646  */
    { threads = dim3((yyvsp[-1]),(yyvsp[0]),1); }
#line 1477 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 26:
#line 141 "matrix.y" /* yacc.c:1646  */
    { tolerance = double_stack[(yyvsp[0])]; }
#line 1483 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 27:
#line 142 "matrix.y" /* yacc.c:1646  */
    { 
			sym[(yyvsp[-2])] = (MATRIX<double> *)malloc(sizeof(MATRIX<double>)+((yyvsp[-1]) * (yyvsp[0]) * sizeof(double)));
			sym[(yyvsp[-2])]->height = (yyvsp[-1]);
			sym[(yyvsp[-2])]->width = (yyvsp[0]);
		}
#line 1493 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 28:
#line 147 "matrix.y" /* yacc.c:1646  */
    { free(sym[(yyvsp[0])]); }
#line 1499 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 29:
#line 148 "matrix.y" /* yacc.c:1646  */
    { 
			int device_size = 0;
			cudaGetDeviceCount(&device_size);
			for (int i = 0; i < device_size; ++i)
			{
				cudaDeviceProp cudaDeviceProp;
				cudaGetDeviceProperties(&cudaDeviceProp, i);
				printf("Running on GPU %d (%s)\n", i, cudaDeviceProp.name); 

				printf("Device has ECC support enabled %d\n",cudaDeviceProp.ECCEnabled);
				printf("Number of asynchronous engines %d\n",cudaDeviceProp.asyncEngineCount);
				printf("Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer %d\n",cudaDeviceProp.canMapHostMemory);
				printf("Clock frequency in kilohertz %d\n",cudaDeviceProp.clockRate);
				printf("Compute mode (See cudaComputeMode) %d\n",cudaDeviceProp.computeMode);
				printf("Device can possibly execute multiple kernels concurrently %d\n",cudaDeviceProp.concurrentKernels);
				printf("Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. %d\n",cudaDeviceProp.deviceOverlap);
				printf("Device is integrated as opposed to discrete %d\n",cudaDeviceProp.integrated);
				printf("Specified whether there is a run time limit on kernels %d\n",cudaDeviceProp.kernelExecTimeoutEnabled);
				printf("Size of L2 cache in bytes %d\n",cudaDeviceProp.l2CacheSize);
				printf("Major compute capability %d\n",cudaDeviceProp.major);
				printf("Maximum size of each dimension of a grid %d\n",cudaDeviceProp.maxGridSize[0]);
				printf("Maximum size of each dimension of a grid %d\n",cudaDeviceProp.maxGridSize[1]);
				printf("Maximum size of each dimension of a grid %d\n",cudaDeviceProp.maxGridSize[2]);
				printf("Maximum 1D surface size %d\n",cudaDeviceProp.maxSurface1D);
				printf("Maximum 1D layered surface dimensions %d\n",cudaDeviceProp.maxSurface1DLayered[0]);
				printf("Maximum 1D layered surface dimensions %d\n",cudaDeviceProp.maxSurface1DLayered[1]);
				printf("Maximum 2D surface dimensions %d\n",cudaDeviceProp.maxSurface2D[0]);
				printf("Maximum 2D surface dimensions %d\n",cudaDeviceProp.maxSurface2D[1]);
				printf("Maximum 2D layered surface dimensions %d\n",cudaDeviceProp.maxSurface2DLayered[0]);
				printf("Maximum 2D layered surface dimensions %d\n",cudaDeviceProp.maxSurface2DLayered[1]);
				printf("Maximum 2D layered surface dimensions %d\n",cudaDeviceProp.maxSurface2DLayered[2]);
				printf("Maximum 3D surface dimensions %d\n",cudaDeviceProp.maxSurface3D[0]);
				printf("Maximum 3D surface dimensions %d\n",cudaDeviceProp.maxSurface3D[1]);
				printf("Maximum 3D surface dimensions %d\n",cudaDeviceProp.maxSurface3D[2]);
				printf("Maximum Cubemap surface dimensions %d\n",cudaDeviceProp.maxSurfaceCubemap);
				printf("Maximum Cubemap layered surface dimensions %d\n",cudaDeviceProp.maxSurfaceCubemapLayered[0]);
				printf("Maximum Cubemap layered surface dimensions %d\n",cudaDeviceProp.maxSurfaceCubemapLayered[1]);
				printf("Maximum 1D texture size %d\n",cudaDeviceProp.maxTexture1D);
				printf("Maximum 1D layered texture dimensions %d\n",cudaDeviceProp.maxTexture1DLayered[0]);
				printf("Maximum 1D layered texture dimensions %d\n",cudaDeviceProp.maxTexture1DLayered[1]);
				printf("Maximum size for 1D textures bound to linear memory %d\n",cudaDeviceProp.maxTexture1DLinear);
				printf("Maximum 1D mipmapped texture size %d\n",cudaDeviceProp.maxTexture1DMipmap);
				printf("Maximum 2D texture dimensions %d\n",cudaDeviceProp.maxTexture2D[0]);
				printf("Maximum 2D texture dimensions %d\n",cudaDeviceProp.maxTexture2D[1]);
				printf("Maximum 2D texture dimensions if texture gather operations have to be performed %d\n",cudaDeviceProp.maxTexture2DGather[0]);
				printf("Maximum 2D texture dimensions if texture gather operations have to be performed %d\n",cudaDeviceProp.maxTexture2DGather[1]);
				printf("Maximum 2D layered texture dimensions %d\n",cudaDeviceProp.maxTexture2DLayered[0]);
				printf("Maximum 2D layered texture dimensions %d\n",cudaDeviceProp.maxTexture2DLayered[1]);
				printf("Maximum 2D layered texture dimensions %d\n",cudaDeviceProp.maxTexture2DLayered[2]);
				printf("Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory %d\n",cudaDeviceProp.maxTexture2DLinear[0]);
				printf("Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory %d\n",cudaDeviceProp.maxTexture2DLinear[1]);
				printf("Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory %d\n",cudaDeviceProp.maxTexture2DLinear[2]);
				printf("Maximum 2D mipmapped texture dimensions %d\n",cudaDeviceProp.maxTexture2DMipmap[0]);
				printf("Maximum 2D mipmapped texture dimensions %d\n",cudaDeviceProp.maxTexture2DMipmap[1]);
				printf("Maximum 3D texture dimensions %d\n",cudaDeviceProp.maxTexture3D[0]);
				printf("Maximum 3D texture dimensions %d\n",cudaDeviceProp.maxTexture3D[1]);
				printf("Maximum 3D texture dimensions %d\n",cudaDeviceProp.maxTexture3D[2]);
				printf("Maximum alternate 3D texture dimensions %d\n",cudaDeviceProp.maxTexture3DAlt[0]);
				printf("Maximum alternate 3D texture dimensions %d\n",cudaDeviceProp.maxTexture3DAlt[1]);
				printf("Maximum alternate 3D texture dimensions %d\n",cudaDeviceProp.maxTexture3DAlt[2]);
				printf("Maximum Cubemap texture dimensions %d\n",cudaDeviceProp.maxTextureCubemap);
				printf("Maximum Cubemap layered texture dimensions %d\n",cudaDeviceProp.maxTextureCubemapLayered[0]);
				printf("Maximum Cubemap layered texture dimensions %d\n",cudaDeviceProp.maxTextureCubemapLayered[1]);
				printf("Maximum size of each dimension of a block %d\n",cudaDeviceProp.maxThreadsDim[0]);
				printf("Maximum size of each dimension of a block %d\n",cudaDeviceProp.maxThreadsDim[1]);
				printf("Maximum size of each dimension of a block %d\n",cudaDeviceProp.maxThreadsDim[2]);
				printf("Maximum number of threads per block %d\n",cudaDeviceProp.maxThreadsPerBlock);
				printf("Maximum resident threads per multiprocessor %d\n",cudaDeviceProp.maxThreadsPerMultiProcessor);
				printf("Maximum pitch in bytes allowed by memory copies %d\n",cudaDeviceProp.memPitch);
				printf("Global memory bus width in bits %d\n",cudaDeviceProp.memoryBusWidth);
				printf("Peak memory clock frequency in kilohertz %d\n",cudaDeviceProp.memoryClockRate);
				printf("Minor compute capability %d\n",cudaDeviceProp.minor);
				printf("Number of multiprocessors on device %d\n",cudaDeviceProp.multiProcessorCount);
				printf("PCI bus ID of the device %d\n",cudaDeviceProp.pciBusID);
				printf("PCI device ID of the device %d\n",cudaDeviceProp.pciDeviceID);
				printf("PCI domain ID of the device %d\n",cudaDeviceProp.pciDomainID);
				printf("32-bit registers available per block %d\n",cudaDeviceProp.regsPerBlock);
				printf("Shared memory available per block in bytes %d\n",cudaDeviceProp.sharedMemPerBlock);
				printf("Device supports stream priorities %d\n",cudaDeviceProp.streamPrioritiesSupported);
				printf("Alignment requirements for surfaces %d\n",cudaDeviceProp.surfaceAlignment);
				printf("1 if device is a Tesla device using TCC driver, 0 otherwise %d\n",cudaDeviceProp.tccDriver);
				printf("Alignment requirement for textures %d\n",cudaDeviceProp.textureAlignment);
				printf("Pitch alignment requirement for texture references bound to pitched memory %d\n",cudaDeviceProp.texturePitchAlignment);
				printf("Constant memory available on device in bytes %d\n",cudaDeviceProp.totalConstMem);
				printf("Global memory available on device in bytes %d\n",cudaDeviceProp.totalGlobalMem);
				printf("Device shares a unified address space with the host %d\n",cudaDeviceProp.unifiedAddressing);
				printf("Warp size in threads %d\n",cudaDeviceProp.warpSize);

				fflush(stdout);
			}
		}
#line 1595 "matrix.tab.cpp" /* yacc.c:1646  */
    break;

  case 30:
#line 239 "matrix.y" /* yacc.c:1646  */
    {
			printf("CUDA matrix calculator\n");
			printf("\thelp\n");
			printf("\tread|write VARIABLE FILENAME\n");
			printf("\tlet VARIABLE = VARIABLE [-+*] VARIABLE\n");
			printf("\tlet VARIABLE = [-+ inv rot gaussjordan] VARIABLE\n");
			printf("\tshow info\n");
			printf("\tuse [src dest cache] [texture constant global shared local none]\n");
			printf("\tset [blocks threads] INTEGER INTEGER\n");
			printf("\tset tolerance DOUBLE\n");
			printf("\talloc VARIABLE INTEGER INTEGER\n");
			printf("\tfree VARIABLE\n");
			fflush(stdout);
		}
#line 1614 "matrix.tab.cpp" /* yacc.c:1646  */
    break;


#line 1618 "matrix.tab.cpp" /* yacc.c:1646  */
      default: break;
    }
  /* User semantic actions sometimes alter yychar, and that requires
     that yytoken be updated with the new translation.  We take the
     approach of translating immediately before every use of yytoken.
     One alternative is translating here after every semantic action,
     but that translation would be missed if the semantic action invokes
     YYABORT, YYACCEPT, or YYERROR immediately after altering yychar or
     if it invokes YYBACKUP.  In the case of YYABORT or YYACCEPT, an
     incorrect destructor might then be invoked immediately.  In the
     case of YYERROR or YYBACKUP, subsequent parser actions might lead
     to an incorrect destructor call or verbose syntax error message
     before the lookahead is translated.  */
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;

  /* Now 'shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*--------------------------------------.
| yyerrlab -- here on detecting error.  |
`--------------------------------------*/
yyerrlab:
  /* Make sure we have latest lookahead translation.  See comments at
     user semantic actions for why this is necessary.  */
  yytoken = yychar == YYEMPTY ? YYEMPTY : YYTRANSLATE (yychar);

  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
# define YYSYNTAX_ERROR yysyntax_error (&yymsg_alloc, &yymsg, \
                                        yyssp, yytoken)
      {
        char const *yymsgp = YY_("syntax error");
        int yysyntax_error_status;
        yysyntax_error_status = YYSYNTAX_ERROR;
        if (yysyntax_error_status == 0)
          yymsgp = yymsg;
        else if (yysyntax_error_status == 1)
          {
            if (yymsg != yymsgbuf)
              YYSTACK_FREE (yymsg);
            yymsg = (char *) YYSTACK_ALLOC (yymsg_alloc);
            if (!yymsg)
              {
                yymsg = yymsgbuf;
                yymsg_alloc = sizeof yymsgbuf;
                yysyntax_error_status = 2;
              }
            else
              {
                yysyntax_error_status = YYSYNTAX_ERROR;
                yymsgp = yymsg;
              }
          }
        yyerror (yymsgp);
        if (yysyntax_error_status == 2)
          goto yyexhaustedlab;
      }
# undef YYSYNTAX_ERROR
#endif
    }



  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
         error, discard it.  */

      if (yychar <= YYEOF)
        {
          /* Return failure if at end of input.  */
          if (yychar == YYEOF)
            YYABORT;
        }
      else
        {
          yydestruct ("Error: discarding",
                      yytoken, &yylval);
          yychar = YYEMPTY;
        }
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  /* Do not reclaim the symbols of the rule whose action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;      /* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (!yypact_value_is_default (yyn))
        {
          yyn += YYTERROR;
          if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
            {
              yyn = yytable[yyn];
              if (0 < yyn)
                break;
            }
        }

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
        YYABORT;


      yydestruct ("Error: popping",
                  yystos[yystate], yyvsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  YY_IGNORE_MAYBE_UNINITIALIZED_BEGIN
  *++yyvsp = yylval;
  YY_IGNORE_MAYBE_UNINITIALIZED_END


  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined yyoverflow || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
    {
      /* Make sure we have latest lookahead translation.  See comments at
         user semantic actions for why this is necessary.  */
      yytoken = YYTRANSLATE (yychar);
      yydestruct ("Cleanup: discarding lookahead",
                  yytoken, &yylval);
    }
  /* Do not reclaim the symbols of the rule whose action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
                  yystos[*yyssp], yyvsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  return yyresult;
}
#line 254 "matrix.y" /* yacc.c:1906  */

void yyerror(char *s) {
	fprintf(stderr, "%s\n", s);
}
int main(void) {

	yyparse();

	cudaDeviceReset();

	while(string_stack_size-->0) free(string_stack[string_stack_size]);
}
