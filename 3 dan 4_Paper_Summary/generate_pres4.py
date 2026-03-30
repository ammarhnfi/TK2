import pandas as pd

df = pd.read_excel('d:/tk2/3_Paper_Summary/Ringkasan_Paper_7-12.xlsx')

tex = r"""\documentclass{beamer}
\usetheme{Madrid}
\usecolortheme{default}

\usepackage{graphicx}
\usepackage{booktabs}

\title[Ringkasan Literatur Spasial 7-12]{RINGKASAN LITERATUR: REGRESI SPASIAL DAN MODEL PANEL}
\subtitle{Paper 7 - 12}
\author[Ammar, Norman, Kirono, Devana]{
    Ammar Hanafi (2206051582) \\ 
    Norman Mowlana Aziz (2206025470) \\ 
    Kirono Dwi Saputro (2106656365) \\ 
    Devana Solea (2306262402)
}
\institute[Univ. Indonesia]{
    Program Studi Sarjana Statistika \\
    Fakultas Matematika dan Ilmu Pengetahuan Alam \\
    Universitas Indonesia
}
\date{Maret 2026}

\begin{document}

\frame{\titlepage}

\begin{frame}[allowframebreaks]{Daftar Isi}
    \tableofcontents
\end{frame}
"""

def escape_tex(text):
    if pd.isna(text):
        return ""
    text = str(text)
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde',
        '^': r'\textasciiticircum',
        '\n': r'\\ '
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

for idx, row in df.iterrows():
    author_year = escape_tex(row['Pengarang & Tahun'])
    title = escape_tex(row['Judul Paper'])
    tujuan = escape_tex(row['Tujuan Penelitian'])
    data_var = escape_tex(row['Data dan Variabel'])
    metode = escape_tex(row['Metode'])
    hasil = escape_tex(row['Hasil dan Kesimpulan'])
    langkah = escape_tex(row['Langkah-langkah Penelitian'])
    simulasi = escape_tex(row['Ada simulasi atau penelitian statistika teori'])
    
    # Slide 1: Judul dan Tujuan
    tex += f"\\section{{{author_year}}}\n"
    tex += f"\\begin{{frame}}[allowframebreaks]{{{author_year}}}\n"
    tex += f"\\textbf{{Judul Paper:}} {title} \\\\[0.3cm]\n"
    tex += f"\\textbf{{Tujuan Penelitian:}} {tujuan} \\\\[0.3cm]\n"
    tex += f"\\textbf{{Metode:}} {metode}\n"
    tex += f"\\end{{frame}}\n\n"
    
    # Slide 2: Data, Variabel, dan Simulasi
    tex += f"\\begin{{frame}}[allowframebreaks]{{Data \& Simulasi: {author_year}}}\n"
    tex += f"\\textbf{{Data dan Variabel:}} {data_var} \\\\[0.3cm]\n"
    tex += f"\\textbf{{Simulasi/Teori:}} {simulasi}\n"
    tex += f"\\end{{frame}}\n\n"

    # Slide 3: Langkah dan Hasil
    tex += f"\\begin{{frame}}[allowframebreaks]{{Hasil: {author_year}}}\n"
    tex += f"\\textbf{{Langkah Penelitian:}} {langkah} \\\\[0.3cm]\n"
    tex += f"\\textbf{{Hasil dan Kesimpulan:}} {hasil}\n"
    tex += f"\\end{{frame}}\n\n"

tex += r"""
\begin{frame}
    \centering
    \Huge \textbf{Terima Kasih}
\end{frame}
\end{document}
"""

with open('d:/tk2/3_Paper_Summary/Presentasi4.tex', 'w', encoding='utf-8') as f:
    f.write(tex)
print("Presentasi4.tex generated successfully.")
