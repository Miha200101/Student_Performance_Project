# importam bibliotecile necesare
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

# citim fisierul CSV local (student-mat.csv)
# specificam separatorul corect ';'
CSV_PATH = "student_performance/student-mat.csv"
df = pd.read_csv(CSV_PATH, sep=';')

# afisam informatii generale despre setul de date
print("Informatii despre setul initial:")
print(df.info(), "\n")
print(df.head(), "\n")

# verificam daca exista valori lipsa (NaN)
print("Verificarea valorilor lipsa:")
print(df.isnull().sum(), "\n")

# nu exista valori lipsa; daca erau, completam media la numerice si moda la text

# in set avem variabile text (F/M, U/R, yes/no, school, guardian, reason)
# le transformam in valori numerice pentru a putea fi folosite ulterior in algoritmi ML

# impartim coloanele text in:
# - binare (2 valori, ex: F/M)
# - multi-clasa (ex: Mjob, reason)
cat_cols   = df.select_dtypes(include="object").columns.tolist()
binary_cols = [c for c in cat_cols if df[c].nunique() == 2]
multi_cols  = [c for c in cat_cols if df[c].nunique() > 2]

# codificam binarele cu LabelEncoder (0/1)
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

# one-hot pentru coloanele multi-clasa (evitam ordine falsa)
# drop_first=True elimina o col/cluster pentru a evita colinearitatea
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

# convertim dummies din bool in 0/1 (optional dar arata mai curat in csv)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype("uint8")

print("Dupa codificare (primele randuri):")
print(df.head(), "\n")

# pregatim listele de coloane pentru scalare
target_cols = ['G1', 'G2', 'G3']                 # nu scalăm tintele
all_feat_cols = [c for c in df.columns if c not in target_cols]

# nu scalăm coloanele binare/dummy (0/1); pastram doar numericele
# comenteaza urmatoarele doua linii daca vrei sa scalezi tot in afara de G1,G2,G3
binary_like = [c for c in all_feat_cols if df[c].dropna().isin([0,1]).all()]
cont_feats  = [c for c in all_feat_cols if c not in binary_like]

# scalare Min-Max (valori intre 0 si 1)
df_minmax = df.copy()
if cont_feats:
    mm = MinMaxScaler()
    df_minmax[cont_feats] = mm.fit_transform(df_minmax[cont_feats])
    df_minmax[cont_feats] = df_minmax[cont_feats].round(3)

print("Dupa scalare Min-Max (primele randuri):")
print(df_minmax.head(), "\n")

# scalare Z-score (media 0, deviatia standard 1)
df_zscore = df.copy()
if cont_feats:
    ss = StandardScaler()
    df_zscore[cont_feats] = ss.fit_transform(df_zscore[cont_feats])

# salvam fisierele rezultate
OUT_MINMAX = "student-mat-preprocessed-minmax.csv"
OUT_ZSCORE = "student-mat-preprocessed-zscore.csv"

try:
    df_minmax.to_csv(OUT_MINMAX, sep=';', index=False, float_format="%.3f", encoding="utf-8")
    df_zscore.to_csv(OUT_ZSCORE, sep=';', index=False, float_format="%.6f", encoding="utf-8")
    print("Fisierele au fost salvate corect.\n")
except PermissionError:
    df_minmax.to_csv("student-mat-preprocessed-minmax-v2.csv", sep=';', index=False, float_format="%.3f", encoding="utf-8")
    df_zscore.to_csv("student-mat-preprocessed-zscore-v2.csv", sep=';', index=False, float_format="%.6f", encoding="utf-8")
    print("Fisierele erau deschise. Am salvat cu alta denumire.\n")

# grafic simplu: distributia notelor finale (G3), nescalata
plt.figure(figsize=(8,5))
df['G3'].hist(bins=20, edgecolor='black')
plt.title("Distributia notelor finale (G3)")
plt.xlabel("Nota finala")
plt.ylabel("Numar de elevi")
plt.tight_layout()
plt.show()

# mici verificari automate
assert df_minmax[target_cols].equals(df[target_cols]), "Tintele au fost modificate; nu este ok."
assert df.isnull().sum().sum() == 0, "Au aparut valori lipsa dupa preprocesare."