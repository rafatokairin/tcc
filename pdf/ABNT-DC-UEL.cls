%%
%% Customizações do abnTeX2 (http://abnTeX2.googlecode.com) para o Departamento de
%% Computação da Universidade Estadual de Londrina
%%
%% This work may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%   http://www.latex-project.org/lppl.txt
%% and version 1.3 or later is part of all distributions of LaTeX
%% version 2005/12/01 or later.
%%
%% This work has the LPPL maintenance status `maintained'.
%% 
%% The Current Maintainer of this work is Daniel dos Santos Kaster (2013-2018) <dskaster@uel.br>
%%
%% Collaborators:
%% Anderson Paulo Ávila Santos (2018) <anderson.avilasantos@gmail.com>
%%
%% Further information about abnTeX2 are available on http://abntex2.googlecode.com/
%%
%% Changelog:
%%
%% 2018/04/02 - (v1.2) Ajustes solicitados pela equipe da Biblioteca Digital da UEL
%% 2014/08/25 - (v1.1) Ajustes diversos de formatação para manter conformidade com template DC/UEL MS Word 2013
%% 2013/10/30 - (v1.0.1) Inclusão das opções tccpreliminar e qualificacaoms
%% 2013/07/11 - (v1.0) Liberação da customização dc-uel.cls do ABNTeX
%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ABNT-DC-UEL}[2018/04/02 v1.2 Modelo da Departamento de Computação da UEL baseado em abnTeX2]

\providecommand{\DCUEListccpreliminar}{false}
\DeclareOption{tccpreliminar}{
	\renewcommand{\DCUEListccpreliminar}{true}
}

\providecommand{\DCUEListcc}{false}
\DeclareOption{tcc}{
	\renewcommand{\DCUEListcc}{true}
}

\providecommand{\DCUELisqualificacaoms}{false}
\DeclareOption{qualificacaoms}{
	\renewcommand{\DCUELisqualificacaoms}{true}
}

\providecommand{\DCUELisdissertacao}{false}
\DeclareOption{dissertacao}{
	\renewcommand{\DCUELisdissertacao}{true}
}

% Passa parâmetros do template DC para a classe abntex2
\PassOptionsToClass{chapter=TITLE}{abntex2} % títulos de capítulos convertidos em letras maiúsculas
\PassOptionsToPackage{english}{babel} % idioma adicional para hifenização/citação

%
% Repassa todos os demais parametros a classe abntex2, 
\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{abntex2}%
}
\ProcessOptions\relax
\LoadClass{abntex2}


% ---
% INICIO DAS CUSTOMIZACOES PARA O DEPARTAMENTO DE COMPUTAÇÃO DA UNIVERSIDADE ESTADUAL DE LONDRINA
% ---

\providecommand{\imprimirmembrobancadois}{}
\newcommand{\membrobancadois}[1]{\renewcommand{\imprimirmembrobancadois}{#1}}

\providecommand{\imprimirmembrobancatres}{}
\newcommand{\membrobancatres}[1]{\renewcommand{\imprimirmembrobancatres}{#1}}

\providecommand{\imprimirmembrobancaquatro}{}
\newcommand{\membrobancaquatro}[1]{\renewcommand{\imprimirmembrobancaquatro}{#1}}

\providecommand{\imprimirinstmembrobancadois}{}
\newcommand{\instmembrobancadois}[1]{\renewcommand{\imprimirinstmembrobancadois}{#1}}

\providecommand{\imprimirinstmembrobancatres}{}
\newcommand{\instmembrobancatres}[1]{\renewcommand{\imprimirinstmembrobancatres}{#1}}

\providecommand{\imprimirinstmembrobancaquatro}{}
\newcommand{\instmembrobancaquatro}[1]{\renewcommand{\imprimirinstmembrobancaquatro}{#1}}

\providecommand{\imprimirdiadefesa}{}
\newcommand{\diadefesa}[1]{\renewcommand{\imprimirdiadefesa}{#1}}

\providecommand{\imprimirnivel}{}
\newcommand{\nivel}[1]{\renewcommand{\imprimirnivel}{#1}}

\providecommand{\imprimircitacaoautor}{}
\newcommand{\citacaoautor}[1]{\renewcommand{\imprimircitacaoautor}{#1}}

\providecommand{\imprimirpalavraschave}{}
\newcommand{\palavraschave}[1]{\renewcommand{\imprimirpalavraschave}{#1}}

\providecommand{\imprimirtituloingles}{}
\newcommand{\tituloingles}[1]{\renewcommand{\imprimirtituloingles}{#1}}

\providecommand{\imprimirpalavraschaveingles}{}
\newcommand{\palavraschaveingles}[1]{\renewcommand{\imprimirpalavraschaveingles}{#1}}

% ---
% Capa
% ---
\renewcommand{\imprimircapa}{%

  % definindo o preambulo
  \ifthenelse{\equal{\DCUEListcc}{true}}
    {\preambulo{Trabalho de Conclus\~{a}o de Curso apresentado ao curso de Bacharelado em Ci\^{e}ncia
      da Computa\c{c}\~{a}o da Universidade Estadual de Londrina para obten\c{c}\~{a}o do t\'{i}tulo de
      Bacharel em Ci\^{e}ncia da Computa\c{c}\~{a}o.}}{}
  \ifthenelse{\equal{\DCUEListccpreliminar}{true}}
    {\preambulo{Vers\~{a}o Preliminar de Trabalho de Conclus\~{a}o de Curso apresentado ao curso de Bacharelado em Ci\^{e}ncia
      da Computa\c{c}\~{a}o da Universidade Estadual de Londrina para obten\c{c}\~{a}o do t\'{i}tulo de
      Bacharel em Ci\^{e}ncia da Computa\c{c}\~{a}o.}}{}
  \ifthenelse{\equal{\DCUELisdissertacao}{true}}
    {\preambulo{Disserta\c{c}\~{a}o apresentada ao Programa de Mestrado em Ci\^{e}ncia  da Computa\c{c}\~{a}o da 
      Universidade Estadual de Londrina para obten\c{c}\~{a}o do t\'{i}tulo de Mestre em Ci\^{e}ncia
      da Computa\c{c}\~{a}o.}}{}
  \ifthenelse{\equal{\DCUELisqualificacaoms}{true}}
    {\preambulo{Qualifica\c{c}\~{a}o apresentada ao Programa de Mestrado em Ci\^{e}ncia  da Computa\c{c}\~{a}o da 
      Universidade Estadual de Londrina como requisito parcial para obten\c{c}\~{a}o do t\'{i}tulo de Mestre em Ci\^{e}ncia
      da Computa\c{c}\~{a}o.}}{}

  \begin{capa}%

    \center
    \includegraphics[width=.7\textwidth]{logo-UEL.jpg}

    \includegraphics[width=1\textwidth]{faixa.pdf}

    \vspace{0.3cm}
    {\large \expandafter\MakeUppercase{\imprimirautor}}

    \begin{DoubleSpace}
    \begin{vplace}[0.6]
    {\ABNTEXchapterfont\bfseries\large \expandafter\MakeUppercase{\imprimirtitulo}}
    \end{vplace}
    \end{DoubleSpace}

    \includegraphics[width=1\textwidth]{faixa.pdf}
    \setlength{\parskip}{0.0cm} 
    {\ABNTEXchapterfont\bfseries \expandafter\MakeUppercase{\imprimirlocal}}
    \par
    {\ABNTEXchapterfont\bfseries \imprimirdata}
  \end{capa}
}

% ---
% Folha de rosto 
% ---
\renewcommand{\folhaderostocontent}{
\begin{center}

    {\large \expandafter\MakeUppercase{\imprimirautor}}

    \begin{vplace}[2]
    
    \begin{DoubleSpace}
    {\ABNTEXchapterfont\bfseries\large \expandafter\MakeUppercase{\imprimirtitulo}}
    \end{DoubleSpace}
    
    \vspace{0.5cm}

 \abntex@ifnotempty{\imprimirpreambulo}{
  \hspace{.45\textwidth}
  \begin{minipage}{.5\textwidth}
  \SingleSpacing
  \imprimirpreambulo

  \vspace*{0.4cm}
  \begin{minipage}[t]{0.25\textwidth}
    {\imprimirorientadorRotulo}
  \end{minipage}
  \hfill
  \begin{minipage}[t]{0.65\textwidth}
      {\imprimirorientador}
  \end{minipage}

  \abntex@ifnotempty{\imprimircoorientador}{%
  \vspace{6pt}
  \begin{minipage}[t]{0.25\textwidth}
    {\imprimircoorientadorRotulo}
  \end{minipage}
  \hfill
  \begin{minipage}[t]{0.65\textwidth}
      {\imprimircoorientador}%
  \end{minipage}
    
  }%

  \end{minipage}%
 }%

    \end{vplace}
 
    \setlength{\parskip}{0.0cm} 
    {\ABNTEXchapterfont\bfseries \expandafter\MakeUppercase{\imprimirlocal}}
    \par
    {\ABNTEXchapterfont\bfseries \imprimirdata}

    
\end{center}
}

% ---
% Folha de aprovação
% ---
\newcommand{\imprimirfolhadeaprovacao}{%
\begin{folhadeaprovacao}

  \begin{center}
    {\large \expandafter\MakeUppercase{\imprimirautor}}

    \begin{DoubleSpace}
    \vspace*{\fill}\vspace*{\fill}
    {\ABNTEXchapterfont\bfseries\large \expandafter\MakeUppercase{\imprimirtitulo}}
    \vspace*{1cm}
    \end{DoubleSpace}

    \hspace{.45\textwidth}
    \begin{minipage}{.5\textwidth}
        \imprimirpreambulo

    \vspace*{1cm}
    \center{\bfseries BANCA EXAMINADORA}

   \assinatura{Orientador: \imprimirorientador \\ \imprimirinstituicao}
   \vspace*{0.4cm}
   \ifthenelse{\not\equal{\imprimirmembrobancadois}{}}{\assinatura{\imprimirmembrobancadois \\ \imprimirinstmembrobancadois}}{}
   \vspace*{0.4cm}
   \ifthenelse{\not\equal{\imprimirmembrobancatres}{}}{\assinatura{\imprimirmembrobancatres \\ \imprimirinstmembrobancatres}}{}
   \vspace*{0.4cm}
   \ifthenelse{\not\equal{\imprimirmembrobancaquatro}{}}{\assinatura{\imprimirmembrobancaquatro \\ \imprimirinstmembrobancaquatro}}{}


   \vspace*{1.5cm}

   \imprimirlocal, \imprimirdiadefesa~de \imprimirdata.

    \end{minipage}%
   \end{center}
      
\end{folhadeaprovacao}
}

% ---
% Resumo
% ---
\renewenvironment{resumo}[1][\resumoname]{%

\begin{SingleSpacing}
   \noindent
   \expandafter\MakeUppercase{\imprimircitacaoautor}. {\bfseries\imprimirtitulo}. \imprimirdata. \pageref*{LastPage}f. 
   \ifthenelse{\equal{\DCUEListcc}{true}}{Trabalho de Conclus\~{a}o de Curso (Bacharelado em Ci\^{e}ncia da Computa\c{c}\~{a}o)}{}%
   \ifthenelse{\equal{\DCUEListccpreliminar}{true}}{Trabalho de Conclus\~{a}o de Curso -- Vers\~{a}o Preliminar (Bacharelado em Ci\^{e}ncia da Computa\c{c}\~{a}o)}{}%
   \ifthenelse{\equal{\DCUELisdissertacao}{true}}{Disserta\c{c}\~{a}o (Mestrado em Ci\^{e}ncia da Computa\c{c}\~{a}o)}{}%
   \ifthenelse{\equal{\DCUELisqualificacaoms}{true}}{Qualifica\c{c}\~{a}o (Mestrado em Ci\^{e}ncia da Computa\c{c}\~{a}o)}{}%
   ~-- \imprimirinstituicao, \imprimirlocal, \imprimirdata.
   \vspace*{1cm}

   \PRIVATEbookmarkthis{#1}
   \renewcommand{\abstractnamefont}{\chaptitlefont}
   \renewcommand{\abstractname}{\ABNTEXchapterupperifneeded{#1}}
   \begin{abstract}
   }{\end{abstract}

  \vspace{\onelineskip}
    
  \noindent
  \textbf{Palavras-chave}: \imprimirpalavraschave
 
\end{SingleSpacing}
  
 \vspace*{\fill}
 \PRIVATEclearpageifneeded

}

% ---
% Abstract
% ---
\newenvironment{Abstract}[1][Abstract]{%

\begin{SingleSpacing}
 
   \noindent
   \expandafter\MakeUppercase{\imprimircitacaoautor}. {\bfseries\imprimirtituloingles}. \imprimirdata. \pageref*{LastPage}p.
   \ifthenelse{\equal{\DCUEListcc}{true}}{Final Project (Bachelor of Science in Computer Science)}{}%
   \ifthenelse{\equal{\DCUEListccpreliminar}{true}}{Final Project -- Draft Version (Bachelor of Science in Computer Science)}{}%
   \ifthenelse{\equal{\DCUELisdissertacao}{true}}{Master's Thesis (Master in Science in Computer Science)}{}%
   \ifthenelse{\equal{\DCUELisqualificacaoms}{true}}{Qualification (Master in Science in Computer Science)}{}%
   ~-- State University of Londrina, \imprimirlocal, \imprimirdata.
   \vspace*{1cm}

   \PRIVATEbookmarkthis{#1}
   \renewcommand{\abstractnamefont}{\chaptitlefont}
   \renewcommand{\abstractname}{\ABNTEXchapterupperifneeded{#1}}
   \begin{abstract}
     \begin{otherlanguage*}{english}
     }{
     \end{otherlanguage*}
   \end{abstract}

  \vspace{\onelineskip}
    
  \noindent
  \textbf{Keywords}: \imprimirpalavraschaveingles

\end{SingleSpacing}

  \vspace*{\fill}
  \PRIVATEclearpageifneeded

}

% ---
% Customização de fontes
% ---
\renewcommand{\ABNTEXchapterfont}{\bfseries} % altera a fonte do capitulo
\renewcommand{\ABNTEXchapterfontsize}{\large} % altera o tamanho da fonte de \chapter (inclusive dos elementos pré-textuais)
\renewcommand{\ABNTEXsectionfontsize}{\large} % altera o tamanho da fonte de \section
\renewcommand{\ABNTEXsubsectionfontsize}{\normalsize} % altera o tamanho da fonte de \subsection

% ---
% PACOTES
% ---

% ---
% Pacotes fundamentais 
% ---

\RequirePackage{cmap}			% Mapear caracteres especiais no PDF
\RequirePackage{lmodern}			% Usa a fonte Latin Modern			
\RequirePackage{lastpage}			% Usado pela Ficha catalográfica
\RequirePackage{indentfirst}		% Indenta o primeiro parágrafo de cada seção.
\RequirePackage{color}			% Controle das cores

% ---
% Pacotes de citações
% ---
\RequirePackage[num, recuo=0.8cm]{abntex2cite}	% Citações numéricas padrão ABNT

% --- 
% CONFIGURAÇÕES DE PACOTES
% --- 

% ---
% Configurações do abntex2cite
% ---
\citebrackets[] % altera as citações no texto para usar colchetes
\renewcommand\@biblabel[1]{{\parbox{0.8cm}{[#1]}}} % modifica a seção de referências para mostrar [1] SOBRENOME, P. em vez de 1 SOBRENOME, P.
% ---

% ---
% Configurações de cabeçalho do pacote memoir
% ---
\makepagestyle{dc-uel-header}
  \makeevenhead{dc-uel-header} %%pagina par
     {\thepage} % esquerda
     {} % centro
     {} % direita
  \makeoddhead{dc-uel-header} %%pagina ímpar ou com oneside
     {} % esquerda
     {} % centro
     {\thepage} % direita
% ---

% ---
% Configurações de aparência do PDF final
% ---
% alterando o aspecto da cor
\definecolor{blue}{RGB}{41,5,195} %Obsolete in version 1.2 of the template
\definecolor{black}{RGB}{0,0,0}

% informações do PDF
\hypersetup{
    brazil,
		pdftitle={\@title}, 
		pdfauthor={\@author},
    	pdfsubject={\imprimirpreambulo},
	    pdfcreator={LaTeX with abnTeX2},
		pdfkeywords={abnt}{latex}{abntex}{abntex2}{trabalho acadêmico}, 
		colorlinks=true,       		% false: boxed links; true: colored links
    	linkcolor=black,          	% color of internal links
    	citecolor=black,        		% color of links to bibliography
    	filecolor=magenta,      		% color of file links
		urlcolor=black,
		bookmarksdepth=4
}

% --- 

% --- 
% Espaçamentos entre linhas e parágrafos 
% --- 

% O tamanho do parágrafo é dado por:
\setlength{\parindent}{1.3cm}

% Controle do espaçamento entre um parágrafo e outro:
\setlength{\parskip}{0.2cm}  % ou \onelineskip

% Constantes do abntex2
\instituicao{Universidade Estadual de Londrina}
\local{Londrina}

% Customização das fontes do sumário
\renewcommand{\cftsubsectionfont}{\normalfont\normalsize}
\renewcommand{\cftsubsectionpagefont}{\cftsubsectionfont}

\renewcommand{\cftsubsubsectionfont}{\normalfont\small}
\renewcommand{\cftsubsubsectionpagefont}{\cftsubsubsectionfont}

% ---
% FIM DAS CUSTOMIZACOES PARA O DEPARTAMENTO DE COMPUTAÇÃO DA UNIVERSIDADE ESTADUAL DE LONDRINA
% ---