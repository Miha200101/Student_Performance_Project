# importam bibliotecile necesare
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# fisierele preprocesate (Tema 2) pentru comparatia min-max vs z-score
preprocessed_files = [
    'student-mat-preprocessed-minmax.csv',
    'student-mat-preprocessed-zscore.csv'
]

# configuram strategia de cross-validation si metricile de evaluare
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
metrics = ['accuracy', 'precision', 'recall', 'f1']

# definim mai multe configuratii pentru RandomForest
configs = [
    {'n_estimators': 50,  'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': None}
]

# rulam experimentul pentru fiecare fisier preprocesat (min-max si z-score)
for file in preprocessed_files:
    print(f"Evaluare RandomForest pentru fisierul: {file}")

    # citim fisierul preprocesat
    data = pd.read_csv(file, sep=';')

    # cream tinta binara: 1 = promovat (G3 >= 10), 0 = nepromovat
    data['target'] = (data['G3'] >= 10).astype(int)

    # eliminam coloanele care nu trebuie folosite la antrenare
    # G1 si G2 contribuie la formarea lui G3 -> eliminam pentru a evita data leakage
    X = data.drop(columns=['G1', 'G2', 'G3', 'target'])
    y = data['target']

    # ne asiguram ca toate valorile sunt numerice
    X = X.apply(pd.to_numeric, errors='coerce')

    results = []

    # rulam 10-fold cross-validation pentru fiecare configuratie
    for params in configs:
        model = RandomForestClassifier(random_state=42, **params)
        print(f"\nConfiguratie: {params}")
        res = {}

        for m in metrics:
            scores = cross_val_score(model, X, y, cv=cv, scoring=m)
            mean_score = scores.mean()
            print(f"{m:>9}: {mean_score:.3f}")
            res[m] = mean_score

        results.append({'config': params, **res})

    # afisam comparativ rezultatele pentru fisierul curent
    print(f"\nRezumat configuratii pentru: {file}")
    for r in results:
        print(r)

    # salvam rezultatele in CSV (optional pentru raport)
    out = pd.DataFrame(results)
    out.to_csv(f"rezultate_clasificare_rf_cv10_{file}.csv", index=False)
    print(f"\n[OK] Rezultate salvate in 'rezultate_clasificare_rf_cv10_{file}.csv'")
