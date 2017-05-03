% LaTeX predložak za pismene radove na
% Fakultetu elektrotehnike i računarstva.
% Verzija: 1.0
%
% Pokriveni radovi su: Seminar, Završni i Diplomski rad
%
% Opcije:
%  - seminar - podesi stil za Seminar (standardna postavka)
%  - diplomski - podesi stil za Diplomski rad
%  - zavrsni - podesi stil za Završni rad
%  - lmodern - koristi font lmodern (standardna postavka)
%  - times - koristi font times
%  - utf8 - učitaj inputenc paket i opcijom utf8 (standardna postavka)
%  - cp1250 - učitaj inputenc paket i opcijom cp1250
%  - authoryear - stil citiranja ``(author, year)'' (standardna postavka)
%  - numeric - stil citiranja ``[broj]''

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{fer}

% Predefinirane vrijednosti opcija
\newif\if@fontlmodern		\global\@fontlmoderntrue
\newif\if@fonttimes			\global\@fonttimesfalse
\newif\if@radzavrsni		\global\@radzavrsnifalse
\newif\if@radseminar		\global\@radseminartrue
\newif\if@raddiplomski		\global\@raddiplomskifalse
\newif\if@citeauthoryear	\global\@citeauthoryeartrue
\newif\if@citenumeric		\global\@citenumericfalse
\newif\if@encutf			\global\@encutftrue
\newif\if@enccp				\global\@enccpfalse

% Postavi font
\DeclareOption{lmodern}{\@fontlmoderntrue%
						\@fonttimesfalse}
\DeclareOption{times}{\@fonttimestrue%
					  \@fontlmodernfalse}
					  
% Tip rada
\DeclareOption{seminar}{\@radseminartrue%
					  \@radzavrsnifalse%
					  \@raddiplomskifalse}
\DeclareOption{zavrsni}{\@radseminarfalse%
					  \@radzavrsnitrue%
					  \@raddiplomskifalse}
\DeclareOption{diplomski}{\@radseminarfalse%
					  \@radzavrsnifalse%
					  \@raddiplomskitrue}
					  
% Encoding
\DeclareOption{utf8}{\@encutftrue%
					 \@enccpfalse}
\DeclareOption{cp1250}{\@encutffalse%
					 \@enccptrue}

% Način citiranja
\DeclareOption{numeric}{\@citeauthoryearfalse%
					    \@citenumerictrue}

\ProcessOptions

\LoadClass[12pt, onecolumn]{report}

\RequirePackage[a4paper, left=3.5cm, right=3cm, bottom=3cm, top=3cm]{geometry}

% Postavljanje encodinga
\if@encutf \RequirePackage[utf8]{inputenc}
\else \if@enccp \RequirePackage[cp1250]{inputenc} \fi
\fi

\RequirePackage[croatian]{babel}
\RequirePackage[T1]{fontenc}
\RequirePackage{graphicx} % Uključeno jer je često korišteno
\RequirePackage{amssymb} % Uključeno jer je često korišteno
\RequirePackage{amsmath} % Uključeno jer je često korišteno
\RequirePackage{fixltx2e}
\RequirePackage{caption}
\RequirePackage{ifthen}
\RequirePackage{url} % Potrebno radi natbiba
\RequirePackage{enumitem} % Potrebno radi izmjene itemize okoline

% Numeriranje literature kod seminara
\RequirePackage[nottoc, numbib]{tocbibind}

% Postavljanje fonta
\if@fonttimes\RequirePackage{times} \fi
\if@fontlmodern\RequirePackage{lmodern} \fi

% Postavljanje stila citiranja
\RequirePackage[numbers, square]{natbib}

% Umjesto poziva \RequirePackage[outsidefoot]{pageno}.
% Neke LaTeX distribucije odbijaju automatski instalirati pageno paket.
% Stoga, dio paketa koji se koristi je kopiran u ovu cls datoteku.
\renewcommand{\ps@plain}{%
   \renewcommand{\@mkboth}{\@gobbletwo}%
   \renewcommand{\@oddhead}{}%
   \renewcommand{\@evenhead}{}%
   \renewcommand{\@evenfoot}{\reset@font\rmfamily\thepage\hfil}%
   \renewcommand{\@oddfoot}{\reset@font\rmfamily\hfil\thepage}}
\pagestyle{plain}

\renewcommand{\captionlabelfont}{\bfseries}
\renewcommand{\captionfont}{\small}

% 1.5 prored, 1.3 je faktor za množenje
\linespread{1.3}

% Promjena naziva poglavlja ``Bibliografija'' -> ``Literatura''
\addto\captionscroatian{%
	\def\bibname{Literatura}
}

% Podešavanje oznaka kod listi (1. razina je crtica, 2. puni krug) 
\renewcommand{\labelitemi}{\textendash}
\renewcommand{\labelitemii}{\textbullet}

% http://www.tex.ac.uk/cgi-bin/texfaq2html?label=seccntfmt
% dodaje tocku nakon section broja ali ne i nakon chapter!
% za chapter se koristi jednostavno http://theoval.cmp.uea.ac.uk/~nlct/latex/thesis/node10.html
\renewcommand*{\@seccntformat}[1]{%
  \csname the#1\endcsname.\quad
}

% http://stackoverflow.com/questions/2426963/modifying-latex-table-of-contents-to-add-a-period-after-chapter-table-figure-numb
% Dodaje u toc tocke
\let \savenumberline \numberline
\def \numberline#1{\savenumberline{#1.}}

\renewcommand\theequation{\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@equation}
\renewcommand\thefigure{\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@figure}
\renewcommand\thetable{\ifnum \c@chapter>\z@ \thechapter.\fi \@arabic\c@table}

% Rimski brojevi stranica za zahvalu i tablicu sadržaja.
\renewcommand{\thepage}{\roman{page}}

\if@radseminar\else%
% Ispis napomene o umetanju izvornika.
\newcommand{\izvornik}{%
	\newpage
	\thispagestyle{empty}
	\vspace*{\fill}
	\hfil \textsl{Umjesto ove stranice umetnite izvornik Va\v{s}eg rada.} \par \hfil
	\hfil \textsl{Da bi ste uklonili ovu stranicu obri\v{s}ite naredbu \texttt{\textbackslash izvornik}.} \hfil
	\vspace*{\fill}
}
\fi

% Dodana zahvala
\newif\if@dodanazahvala		\global\@dodanazahvalafalse
\if@radseminar\else%
% Naredba za dodavanje zahvale.
\newcommand{\zahvala}[1]{%
	\newpage
 	\setcounter{page}{3}
	\vspace*{\fill}
	\hfil {\itshape #1}\hfil
	\vspace*{\fill}
	\@dodanazahvalatrue
}
\fi

% TOC
\renewcommand\tableofcontents{%
   \if@dodanazahvala\else \setcounter{page}{4} \fi
   \if@radseminar \setcounter{page}{2} \fi
   \chapter*{\contentsname
   \@mkboth{%
   \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
   \@starttoc{toc}%
}

\newboolean{atappendix}
\setboolean{atappendix}{false}
\newboolean{secondpart}
\setboolean{secondpart}{false}

\renewcommand{\appendix}{
	\setboolean{atappendix}{true}
	\setcounter{chapter}{0}
	\renewcommand{\thechapter}{\Alph{chapter}}
}

\renewcommand{\@makechapterhead}[1]{%
\ifthenelse{\boolean{atappendix}}{%
	\vspace*{50\p@}%
	{\setlength{\parskip}{0em} \parindent \z@ \raggedright \normalfont
		\interlinepenalty\@M
		\ifnum \c@secnumdepth >\m@ne
			\Huge\bfseries Dodatak \thechapter \\ #1
			\par\nobreak
		\fi
		\nobreak
		\vskip 40\p@
	}
}
{
	\ifthenelse{\boolean{secondpart}}
	{
	}
	{
		\setcounter{page}{1}
		\renewcommand{\thepage}{\arabic{page}}
		\setboolean{secondpart}{true}
	}

	\vspace*{50\p@}%
	{\setlength{\parskip}{0em} \parindent \z@ \raggedright \normalfont
		\interlinepenalty\@M
		\ifnum \c@secnumdepth >\m@ne
			\Huge\bfseries \thechapter.~#1
			\par\nobreak
		\fi
		\nobreak
		\vskip 40\p@
	}
}
}

\renewcommand{\@makeschapterhead}[1]{%
\ifthenelse{\boolean{atappendix}}{%
	\vspace*{50\p@}%
	{\setlength{\parskip}{0em} \parindent \z@ \raggedright
		\normalfont
		\interlinepenalty\@Mdef
		\Huge\scshape Dodatak \\ #1\par
		\nobreak
		\vskip 40\p@
	}
}
{%
	\vspace*{50\p@}%
	{\setlength{\parskip}{0em} \parindent \z@ \raggedright
		\normalfont
		\interlinepenalty\@M
		\Huge\scshape #1\par
		\nobreak
		\vskip 40\p@
	}
}
}

\if@radseminar%
  \def\voditelj#1{\gdef\@voditelj{#1}}
  \def\@voditelj{\@latex@warning@no@line{Voditelj nije naveden.
  Koristi \noexpand\voditelj za definiranje}}
\else%
  \def\thesisnumber#1{\gdef\@thesisnumber{#1}}
  \def\@thesisnumber{\@latex@warning@no@line{Broj rada nije definiran.
  Koristi \noexpand\thesisnumber za definiranje}}
\fi

\if@titlepage
	\renewcommand\maketitle{%
	\begin{titlepage}%
		\let\footnotesize\small
		\let\footnoterule\relax
	
		\begin{center}
			{\large\sffamily SVEU\v{C}ILI\v{S}TE U ZAGREBU}
			
			{\large\bfseries\sffamily FAKULTET ELEKTROTEHNIKE I RA\v{C}UNARSTVA}
		\end{center}
		\vfill
		\begin{center}%
	    	\if@radzavrsni%
			  {\large\sffamily ZAVR\v{S}NI RAD br.~\@thesisnumber \par}%
			\else \if@raddiplomski%
			  {\large\sffamily DIPLOMSKI RAD br.~\@thesisnumber \par}%
			\else {\large\sffamily\bfseries SEMINAR \par}%
			\fi\fi
			\if@radseminar \vskip 3em%
			\else \vskip 1em%
			\fi
			{\huge \bfseries\sffamily \@title \par}%
			\vskip 1em%
			{\large\sffamily\lineskip .75em%
			\begin{tabular}[t]{c}%
	        	\if@radseminar%
	        	  {\itshape \@author} \\
	        	  Voditelj: {\itshape \@voditelj}
	        	\else
				    \@author
				\fi
			\end{tabular}\par}%
		\end{center}\par
		\vfill
		\begin{center}
			{\sffamily\large Zagreb, 
			\ifcase\month\or
				sije\v{c}anj\or velja\v{c}a\or o\v{z}ujak\or travanj\or svibanj\or
	   	 		lipanj\or srpanj\or kolovoz\or rujan\or listopad\or studeni\or
	    			prosinac\fi \space \number\year.%
	    	}
		\end{center}
	\end{titlepage}%
	\setcounter{footnote}{0}%
	\global\let\maketitle\relax
% 	\global\let\@thanks\@empty
% 	\global\let\@author\@empty
	\global\let\@date\@empty
%	\global\let\@title\@empty
%	\global\let\title\relax
% 	\global\let\author\relax
	\global\let\date\relax
	\global\let\and\relax
	}
\fi

\newdimen\listindentation
\setlength{\listindentation}{3 em}

% Podešavanje izemize okoline
% Zahtjeva enumitem paket
\renewenvironment{itemize}%
{
	\setlength{\parskip}{0em}
	\advance\@itemdepth\@ne
	\edef\@itemitem{labelitem\romannumeral\@itemdepth}%

	\begin{list}{\csname\@itemitem\endcsname}{\setlength{\leftmargin}{\listindentation}\setlength{\itemsep}{0 em}\setlength{\topsep}{0 em}}
}%
{\end{list}}

% Naredba \engl
\newcommand{\engl}[1]{(engl.~\emph{#1})}

% Sažetak na hrvatskom
\newenvironment{sazetak}
{
\newpage
\vspace*{\fill}
\thispagestyle{empty}
\begin{center}
  {\bf \@title}
\end{center}
\hspace*{\fill} {\bf Sa\v{z}etak} \hspace*{\fill} \par
\vspace*{25pt}
}
{
\vspace*{\fill}
}

% Ključne riječi na hrvatskom
\newcommand{\kljucnerijeci}[1]
{
\vspace{15pt}
\noindent \textbf{Klju\v{c}ne rije\v{c}i:} #1
}

% Ključne riječi na engleskom
\newcommand{\keywords}[1]
{
\vspace{15pt}
\noindent \textbf{Keywords:} #1
}

% Sažetak na engleskom
\def\engtitle#1{\gdef\@engtitle{#1}}
\def\@engtitle{\@latex@warning@no@line{Engleski naziv rada nije definiran.
Koristi \noexpand\engtitle za definiranje}}
\renewenvironment{abstract}
{
\vspace*{\fill}
\thispagestyle{empty}
\begin{center}
  {\bf \@engtitle}
\end{center}
\hspace*{\fill} {\bf Abstract} \hspace*{\fill} \par
\vspace*{25pt}
}
{
\vspace*{\fill}
}

\endinput
