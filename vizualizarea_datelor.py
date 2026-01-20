import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Incarcam setul de date (prelucrat la tema 2)
df = pd.read_csv(r"student_performance\student-mat.csv", sep=";")

# Pregatim datele pentru grafice
# media notei G3 pe fiecare nivel de studytime
mean_g3 = (
    df.groupby("studytime", as_index=False)["G3"]
    .mean()
    .rename(columns={"G3": "G3_mean"})
)

# Cream figura cu 3 grafice
fig = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=(
        "Distributia notei finale G3",
        "Media notei G3 in functie de studytime",
        "Relatia dintre absente si G3",
    ),
    horizontal_spacing=0.08,
)

# Grafic 1 - Histograma pentru G3
hist = px.histogram(
    df,
    x="G3",
    nbins=15,
    color_discrete_sequence=["#636EFA"],
)
hist_tr = hist.data[0]
hist_tr.showlegend = False
fig.add_trace(hist_tr, row=1, col=1)

fig.update_xaxes(title_text="Nota finala G3", row=1, col=1)
fig.update_yaxes(title_text="Numar de elevi", row=1, col=1)

# Grafic 2 - media lui G3 pe studytime (BAR CORECT)
bar = px.bar(
    mean_g3,
    x="studytime",
    y="G3_mean",
    text="G3_mean",
    color_discrete_sequence=["#00CC96"],
)
bar_tr = bar.data[0]
bar_tr.showlegend = False
bar_tr.texttemplate = "%{text:.1f}"
bar_tr.textposition = "outside"
fig.add_trace(bar_tr, row=1, col=2)

fig.update_xaxes(title_text="Studytime (nivel 1 - 4)", row=1, col=2)
fig.update_yaxes(title_text="Media notei G3", row=1, col=2)

# Grafic 3 - Scatter absente vs G3 + curba LOWESS
scatter = px.scatter(
    df,
    x="absences",
    y="G3",
    trendline="lowess",
    opacity=0.7,
    color_discrete_sequence=["#EF553B"],
)
for tr in scatter.data:
    tr.showlegend = False
    fig.add_trace(tr, row=1, col=3)

fig.update_xaxes(title_text="Numar de absente", row=1, col=3)
fig.update_yaxes(title_text="Nota finala G3", row=1, col=3)

# Setari finale ale raportului
fig.update_layout(
    title="Raport vizualizare Student Performance",
    template="plotly_white",
    width=1600,
    height=550,
    font=dict(family="Arial", size=12),
    margin=dict(l=40, r=40, t=80, b=40),
)

# Salvam raportul in fisier HTML si il deschidem direct
fig.write_html(
    "raport_vizualizare_student_performance.html",
    include_plotlyjs="inline",
    auto_open=True,
)