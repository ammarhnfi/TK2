import fitz
import os

pdf_files = [
    '1_Geographically_Weighted_Poisson_Regression_GWPR_Mo.pdf',
    '2_EBSCO-FullText-03_01_2026.pdf',
    '3_Spatial Autoregressive Models for Geographically Hierarchical Data Structures.pdf',
    '4_Random effects specifications in eigenvector spatial.pdf',
    '5_Geographically weighted regression with a non-Euclidean distance metric  a case study using hedonic house price data.pdf',
    '6_How to Interpret the Coefficients of Spatial Models.pdf'
]

for idx, pdf_file in enumerate(pdf_files):
    if not os.path.exists(pdf_file):
        print(f"File not found: {pdf_file}")
        continue
    try:
        doc = fitz.open(pdf_file)
        text = ""
        for page in doc:
            text += page.get_text()
        
        out_name = f"paper{idx+1}_extracted.txt"
        with open(out_name, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Extracted {pdf_file} to {out_name}")
    except Exception as e:
        print(f"Error processing {pdf_file}: {e}")
