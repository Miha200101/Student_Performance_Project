# importam bibliotecile necesare
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# fisierele preprocesate obtinute in Tema 2
preprocessed_files = [
    "student-mat-preprocessed-minmax.csv",
    "student-mat-preprocessed-zscore.csv"
]

# valorile de K pe care le testam
k_values = [2, 3, 4]

# parcurgem fiecare fisier preprocesat
for file in preprocessed_files:
    print(f"\nClustering K-means pentru fisierul: {file}")

    # incarcam setul de date
    data = pd.read_csv(file, sep=";")

    # pregatirea datelor: eliminam notele G1, G2, G3
    X = data.drop(columns=["G1", "G2", "G3"])

    # curatam formatul numeric (separatori de mii si virgula zecimala)
    X = X.apply(
        lambda col: col.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )

    # convertim totul la float
    X_values = X.values.astype(float)
    print("Dimensiune set de date:", X_values.shape)

    # lista pentru rezultate pe fiecare K
    results = []
    best_sil = -1.0
    best_model = None

    # bucla peste valorile K
    for k in k_values:
        print(f"\nRulare K-means cu K = {k}")

        # definim modelul K-means (initializare random, ca in articol)
        model = KMeans(
            n_clusters=k,
            init="random",      # alegem centroizi initiali random
            n_init=10,          # rulam de mai multe ori cu porniri diferite
            max_iter=300,       # numar maxim de iteratii
            random_state=42     # seed fix pentru reproductibilitate
        )

        # antrenam modelul si obtinem etichetele de cluster
        labels = model.fit_predict(X_values)

        # inertia_ reprezinta WCSS (sum of squared distances in interiorul clusterelor)
        wcss = model.inertia_

        # calculam Silhouette Score pentru evaluare
        sil = -1.0
        if k > 1:
            sil = silhouette_score(X_values, labels)

        print("Silhouette score:", sil)
        print("WCSS:", wcss)

        # salvam rezultatele pentru acest K
        results.append({
            "k": k,
            "wcss": wcss,
            "silhouette": sil
        })

        # retinem modelul cu Silhouette maxim
        if sil > best_sil:
            best_sil = sil
            best_model = {
                "k": k,
                "labels": labels
            }

    # salvam metricele pentru fiecare K in fisier CSV
    results_df = pd.DataFrame(results)
    out_metrics = f"rezultate_clustering_kmeans_{file}.csv"
    results_df.to_csv(out_metrics, index=False)
    print("Rezultate metrice salvate in:", out_metrics)

    # adaugam clusterul cel mai bun in dataset si il salvam
    if best_model is not None:
        k_best = best_model["k"]
        labels_best = best_model["labels"]

        data_out = data.copy()
        data_out["cluster_kmeans"] = labels_best

        out_clusters = f"clustere_kmeans_K{k_best}_{file}"
        data_out.to_csv(out_clusters, sep=";", index=False)
        print("Clustere finale salvate in:", out_clusters)
