import random
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from surprise import Dataset, KNNBasic, Reader, SVD
from collections import defaultdict
import pandas as pd

st.set_page_config(page_title="Reco Mentors", page_icon="🎓")

st.title(" Système de recommandation de mentors")

# Charger les données
df = pd.read_csv("Parent_Mentor_Matrix.csv")
df_long = df.reset_index().melt(id_vars='parent_id', var_name='mentor_id', value_name='rating')
df_long = df_long[df_long['rating'] > 0]

# Préparation Surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_long[['parent_id', 'mentor_id', 'rating']], reader)
trainset = data.build_full_trainset()

# Sélection du modèle
model_choice = st.radio("Choisir le modèle de recommandation :", ["SVD", "kNN", "Blend (SVD + kNN)"])


algo = SVD()
algo.fit(trainset)

# Interface
parent_ids = df_long["parent_id"].unique()
selected_parent = st.selectbox("Choisir un parent :", parent_ids)

# Recommandations
all_mentors = df_long["mentor_id"].unique()
parent_data = df_long[df_long['parent_id'] == selected_parent]
seen_mentors = parent_data['mentor_id'].tolist()

# Simulation : on suppose que le parent n’a vu que 70% des mentors qu’il a notés
if len(seen_mentors) >= 3:  # pour éviter les erreurs si trop peu de mentors
    masked_seen = random.sample(seen_mentors, int(0.7 * len(seen_mentors)))
else:
    masked_seen = seen_mentors

# Mentors à recommander (non simulés comme vus)
non_interacted = [m for m in all_mentors if m not in masked_seen]

# 🧠 Entraînement des modèles
algo_svd = SVD()
algo_knn = KNNBasic(sim_options={"user_based": False})
algo_svd.fit(trainset)
algo_knn.fit(trainset)

# 🧪 Recommandation selon le modèle sélectionné
if model_choice == "SVD":
    preds = [(m, algo_svd.predict(selected_parent, m).est) for m in non_interacted]
elif model_choice == "kNN":
    preds = [(m, algo_knn.predict(selected_parent, m).est) for m in non_interacted]
elif model_choice == "Blend (SVD + kNN)":
    alpha = 0.5  # pondération
    preds = []
    for m in non_interacted:
        svd_pred = algo_svd.predict(selected_parent, m).est
        knn_pred = algo_knn.predict(selected_parent, m).est
        blend_score = alpha * svd_pred + (1 - alpha) * knn_pred
        preds.append((m, blend_score))


# Affichage des recommandations
st.subheader("Recommandations simulées")
if preds:
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)[:5]
    for mentor, score in preds_sorted:
        st.write(f"Mentor **{mentor}** – Score estimé : `{score:.2f}`")
    
        
    mentors, scores_blend = zip(*preds_sorted)
    scores_svd = [algo_svd.predict(selected_parent, m).est for m in mentors]
    scores_knn = [algo_knn.predict(selected_parent, m).est for m in mentors]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(mentors))
    width = 0.25

    ax.bar(x - width, scores_svd, width, label='SVD', color='#6AC6E8')
    ax.bar(x, scores_knn, width, label='kNN', color='#F28C38')
    ax.bar(x + width, scores_blend, width, label='Blend', color='#888888')

    ax.set_ylabel('Score estimé')
    ax.set_title('Comparaison des scores des mentors recommandés')
    ax.set_xticks(x)
    ax.set_xticklabels(mentors)
    ax.legend()

    st.pyplot(fig)

else:
    st.warning("Aucun mentor à recommander pour ce parent. Tous les mentors sont déjà notés.")

with st.expander("Analyse comparative des modèles", expanded=False):
    st.markdown("""
    **Comparaison des performances**

    | Modèle | RMSE | MAE |
    |--------|------|-----|
    | SVD | 0.5168 | 0.4203 |
    | kNN | 0.4897 | 0.3976 |
    | Blend (α = 0.70) | 0.4959 | 0.4029 |

    **Interprétation :**
    - Le modèle **SVD** prédit toujours des scores proches de 5.00, sans variation selon le parent → peu utile.
    - Le **kNN** propose des scores **personnalisés et cohérents** selon le parent → plus réaliste.
    - Le **Blend** combine les deux approches pour fournir des recommandations **équilibrées et fiables**.

    En conclusion, le modèle blendé est **le plus adapté** pour une expérience utilisateur pertinente.
    """)
