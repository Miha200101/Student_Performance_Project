# Student Performance Project

Acest repository contine proiectul final realizat pentru disciplina
"Modelarea si analiza datelor pentru decizii de management".

Proiectul are ca scop analiza performantei elevilor folosind tehnici
de preprocesare a datelor, clasificare, clustering si vizualizare
interactiva a datelor.

---

## Setul de date

Setul de date utilizat este Student Performance si contine informatii
despre elevi, precum:
- date demografice
- absente
- timp de studiu
- note scolare (G1, G2, G3)

Nota finala G3 este folosita ca variabila principala de analiza.

---

## Structura proiectului

- preprocesarea_datelor.py  
  Curatarea si transformarea datelor, encoding pentru variabilele
  categorice si scalare folosind Min-Max si Z-score.

- clasificare.py  
  Clasificarea elevilor in promovat / nepromovat folosind
  Random Forest Classifier si evaluare prin 10-fold cross-validation.

- kmeans_clustering.py  
  Aplicarea algoritmului K-means pentru identificarea grupurilor
  de elevi, evaluata cu WCSS si Silhouette Score.

- vizualizarea_datelor.py  
  Generarea unui raport interactiv folosind Plotly, cu:
  - histograma pentru distributia notei G3
  - bar chart pentru media lui G3 in functie de studytime
  - scatter plot pentru relatia dintre absente si G3

- raport_vizualizare_student_performance.html  
  Raport interactiv generat automat cu Plotly.

- Raport Proiect_Student Performance.pdf  
  Documentatia finala a proiectului.

- student_performance/  
  Folder care contine fisierele CSV originale ale setului de date.

---

## Metode utilizate

- Preprocesare date:
  - Label Encoding
  - One-Hot Encoding
  - Scalare Min-Max si Z-score

- Clasificare:
  - Random Forest Classifier
  - 10-fold cross-validation
  - metrici: accuracy, precision, recall, F1-score

- Clustering:
  - K-means
  - evaluare folosind WCSS si Silhouette Score

- Vizualizare:
  - Plotly (rapoarte interactive HTML)

---

## Observatii finale

Rezultatele arata ca timpul de studiu are un impact mai clar asupra
performantei elevilor decat numarul de absente. Clasificarea a oferit
rezultate stabile, iar clustering-ul a evidentiat existenta mai multor
profiluri de elevi in functie de comportamentul scolar.

---

## Autor

Miha200101
