"""
=============================================================================
  ACM CRÉDIT BANCAIRE — APPLICATION STREAMLIT
  Analyse des Correspondances Multiples pour le Profiling du Risque Crédit
=============================================================================
  Auteur   : [Votre Nom]
  Stack    : Python · prince · sklearn · plotly · streamlit
  Usage    : streamlit run app_acm_credit.py
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import prince
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACM — Risque Crédit Bancaire",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un rendu professionnel
st.markdown("""
<style>
    /* Fond général */
    .main { background-color: #F8F9FA; }
    
    /* Titres */
    h1 { color: #1A237E; font-size: 2.2rem !important; }
    h2 { color: #283593; border-bottom: 2px solid #3F51B5; padding-bottom: 8px; }
    h3 { color: #3949AB; }
    
    /* Métriques */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #3F51B5;
        margin: 8px 0;
    }
    
    /* Badges de risque */
    .badge-bon { background: #E8F5E9; color: #2E7D32; border-radius: 12px; padding: 4px 12px; font-weight: bold; }
    .badge-moyen { background: #FFF3E0; color: #E65100; border-radius: 12px; padding: 4px 12px; font-weight: bold; }
    .badge-mauvais { background: #FFEBEE; color: #C62828; border-radius: 12px; padding: 4px 12px; font-weight: bold; }
    
    /* Info boxes */
    .info-box {
        background: #E3F2FD;
        border-left: 4px solid #1976D2;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 10px 0;
    }
    
    /* Sidebar */
    .css-1d391kg { background-color: #1A237E; }
    
    /* Séparateur stylé */
    hr { border: none; border-top: 2px solid #E8EAF6; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATION DU DATASET (mis en cache pour performance)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=1000, seed=42):
    """Génère un dataset réaliste de crédit bancaire."""
    np.random.seed(seed)
    profils = np.random.choice(['A', 'B', 'C'], size=n, p=[0.40, 0.35, 0.25])
    
    def gen_var(profil_array, choices_dict):
        return np.array([
            np.random.choice(choices_dict[p][0], p=choices_dict[p][1])
            for p in profil_array
        ])
    
    df = pd.DataFrame({
        'Marche': gen_var(profils, {
            'A': (['Positif', 'Positif_fort', 'Sans_compte'], [0.50, 0.40, 0.10]),
            'B': (['Positif', 'Nul', 'Débiteur_faible'], [0.40, 0.35, 0.25]),
            'C': (['Débiteur_faible', 'Débiteur_fort', 'Nul'], [0.40, 0.35, 0.25])
        }),
        'Apport': gen_var(profils, {
            'A': (['Elevé', 'Moyen', 'Faible'], [0.45, 0.40, 0.15]),
            'B': (['Moyen', 'Faible', 'Elevé'], [0.45, 0.40, 0.15]),
            'C': (['Faible', 'Très_faible', 'Moyen'], [0.50, 0.30, 0.20])
        }),
        'Impaye': gen_var(profils, {
            'A': (['Aucun', 'Ponctuel'], [0.85, 0.15]),
            'B': (['Aucun', 'Ponctuel', 'Répété'], [0.55, 0.30, 0.15]),
            'C': (['Répété', 'Ponctuel', 'Aucun'], [0.50, 0.30, 0.20])
        }),
        'Assurance': gen_var(profils, {
            'A': (['Décès_invalidité', 'Complète', 'Aucune'], [0.50, 0.35, 0.15]),
            'B': (['Décès_invalidité', 'Aucune', 'Complète'], [0.45, 0.35, 0.20]),
            'C': (['Aucune', 'Décès_invalidité', 'Complète'], [0.55, 0.30, 0.15])
        }),
        'Endettement': gen_var(profils, {
            'A': (['Faible', 'Modéré', 'Elevé'], [0.55, 0.35, 0.10]),
            'B': (['Modéré', 'Elevé', 'Faible'], [0.50, 0.30, 0.20]),
            'C': (['Elevé', 'Très_élevé', 'Modéré'], [0.45, 0.35, 0.20])
        }),
        'Famille': gen_var(profils, {
            'A': (['Marié', 'Célibataire', 'Divorcé'], [0.55, 0.30, 0.15]),
            'B': (['Célibataire', 'Marié', 'Divorcé'], [0.45, 0.35, 0.20]),
            'C': (['Célibataire', 'Divorcé', 'Marié'], [0.45, 0.35, 0.20])
        }),
        'Enfants': gen_var(profils, {
            'A': (['0', '1-2', '3+'], [0.25, 0.55, 0.20]),
            'B': (['0', '1-2', '3+'], [0.40, 0.45, 0.15]),
            'C': (['0', '3+', '1-2'], [0.35, 0.35, 0.30])
        }),
        'Logement': gen_var(profils, {
            'A': (['Propriétaire', 'Locataire', 'Hébergé'], [0.55, 0.35, 0.10]),
            'B': (['Locataire', 'Propriétaire', 'Hébergé'], [0.50, 0.30, 0.20]),
            'C': (['Locataire', 'Hébergé', 'Propriétaire'], [0.50, 0.30, 0.20])
        }),
        'Profession': gen_var(profils, {
            'A': (['Cadre', 'Employé_qualifié', 'Fonctionnaire'], [0.35, 0.40, 0.25]),
            'B': (['Employé_qualifié', 'Ouvrier', 'Cadre'], [0.40, 0.35, 0.25]),
            'C': (['Ouvrier', 'Sans_emploi', 'Indépendant'], [0.35, 0.35, 0.30])
        }),
        'Intitule': gen_var(profils, {
            'A': (['Immobilier', 'Voiture_neuve', 'Consommation'], [0.45, 0.35, 0.20]),
            'B': (['Voiture_neuve', 'Consommation', 'Immobilier'], [0.40, 0.35, 0.25]),
            'C': (['Consommation', 'Voiture_occasion', 'Voiture_neuve'], [0.45, 0.30, 0.25])
        }),
        'Age': [int(np.clip(np.random.normal(*{'A':(42,10),'B':(35,8),'C':(30,9)}[p]), 20, 75)) for p in profils],
        'Profil_latent': profils,
        'Risque': np.where(profils=='A', 'Bon', np.where(profils=='B', 'Moyen', 'Mauvais'))
    })
    
    df['Age_groupe'] = pd.cut(df['Age'], bins=[19,29,39,49,59,100],
                               labels=['20-29','30-39','40-49','50-59','60+'])
    return df

@st.cache_data
def run_mca(_df, acm_vars, n_components=8, seed=42):
    """Lance l'ACM et retourne les résultats."""
    X = _df[acm_vars].copy()
    mca = prince.MCA(n_components=n_components, n_iter=10,
                     random_state=seed, engine='sklearn')
    mca.fit(X)
    row_coords = mca.row_coordinates(X)
    col_coords = mca.column_coordinates(X)
    row_coords.columns = [f'Axe_{i+1}' for i in range(row_coords.shape[1])]
    col_coords.columns = [f'Axe_{i+1}' for i in range(col_coords.shape[1])]
    eigs = list(mca.eigenvalues_)
    total = sum(eigs)
    pct = [round(e/total*100, 2) for e in eigs]
    cumul = [round(sum(pct[:i+1]), 2) for i in range(len(pct))]
    return mca, row_coords, col_coords, eigs, pct, cumul

@st.cache_data
def compute_cramer(_df, cat_cols):
    """Calcule la matrice des V de Cramér."""
    def cv(x, y):
        ct = pd.crosstab(x, y)
        chi2, p, dof, exp = chi2_contingency(ct)
        n = ct.sum().sum()
        r, k = ct.shape
        phi2 = max(0, chi2/n - (k-1)*(r-1)/(n-1))
        r_c = r - (r-1)**2/(n-1)
        k_c = k - (k-1)**2/(n-1)
        v = np.sqrt(phi2/min(r_c-1, k_c-1)) if min(r_c-1,k_c-1)>0 else 0
        return round(v, 3)
    mat = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    for c1 in cat_cols:
        for c2 in cat_cols:
            mat.loc[c1,c2] = 1.0 if c1==c2 else cv(_df[c1], _df[c2])
    return mat.astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# BARRE LATÉRALE
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 ACM Crédit Bancaire")
    st.markdown("---")
    
    st.markdown("### ⚙️ Paramètres")
    
    n_clients = st.slider("Nombre de clients", 200, 2000, 1000, 100)
    seed = st.number_input("Graine aléatoire", 0, 999, 42)
    
    st.markdown("### 🔬 Variables ACM")
    all_vars = ['Marche', 'Apport', 'Impaye', 'Assurance', 'Endettement',
                'Famille', 'Enfants', 'Logement', 'Profession', 'Intitule']
    acm_vars_selected = st.multiselect(
        "Variables actives :", all_vars,
        default=['Marche', 'Assurance', 'Intitule', 'Impaye', 'Profession', 'Endettement']
    )
    
    st.markdown("### Clustering")
    k_clusters = st.slider("Nombre de clusters K", 2, 6, 3)
    
    st.markdown("### 🗺️ Plan Factoriel")
    axe_x = st.selectbox("Axe X", [1,2,3,4], index=0)
    axe_y = st.selectbox("Axe Y", [1,2,3,4], index=1)
    
    st.markdown("---")
    st.markdown("""
    **Stack technique :**
    - Python 3.11
    - `prince` (MCA)
    - `sklearn` (K-Means)
    - `plotly` (visualisation)
    - `streamlit` (app)
    
    ---
    *Projet Portfolio — CV*
    """)

# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
if len(acm_vars_selected) < 2:
    st.warning("⚠️ Sélectionnez au moins 2 variables pour l'ACM.")
    st.stop()

data = generate_dataset(n=n_clients, seed=seed)
mca_model, row_c, col_c, eigs, pct_var, cumul_var = run_mca(data, acm_vars_selected)
cramer_mat = compute_cramer(data, acm_vars_selected)

# Clustering
X_cl = row_c.iloc[:, :min(3, row_c.shape[1])].values
km = KMeans(n_clusters=k_clusters, random_state=42, n_init=20)
data['Cluster'] = km.fit_predict(X_cl)
data['ACM_Axe1'] = row_c['Axe_1'].values
data['ACM_Axe2'] = row_c['Axe_2'].values

ax_x_col = f'Axe_{axe_x}'
ax_y_col = f'Axe_{axe_y}'
if ax_x_col not in row_c.columns:
    ax_x_col = 'Axe_1'
if ax_y_col not in row_c.columns:
    ax_y_col = 'Axe_2'

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🏦 Analyse ACM — Risque Crédit Bancaire")
st.markdown("*Analyse des Correspondances Multiples pour le Profiling Client*")

tabs = st.tabs([
    "📊 Vue d'ensemble",
    "🔗 Associations",
    "⚡ Valeurs Propres",
    "🗺️ Plan Factoriel",
    "👥 Clustering",
    "🔮 Scoring Client"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 : VUE D'ENSEMBLE
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("##  Vue d'Ensemble du Portefeuille")
    
    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    n_total = len(data)
    n_bon = (data['Risque']=='Bon').sum()
    n_moyen = (data['Risque']=='Moyen').sum()
    n_mauvais = (data['Risque']=='Mauvais').sum()
    
    col1.metric("👥 Total clients", f"{n_total:,}")
    col2.metric("🟢 Risque Bon", f"{n_bon:,}", f"{n_bon/n_total*100:.1f}%")
    col3.metric("🟡 Risque Moyen", f"{n_moyen:,}", f"{n_moyen/n_total*100:.1f}%")
    col4.metric("🔴 Risque Mauvais", f"{n_mauvais:,}", f"{n_mauvais/n_total*100:.1f}%")
    col5.metric("📈 Variables ACM", len(acm_vars_selected))
    
    st.markdown("---")
    
    # Distributions
    st.markdown("### Distribution des Variables Catégorielles")
    
    vars_to_show = acm_vars_selected
    n_cols = 3
    n_rows = (len(vars_to_show) + n_cols - 1) // n_cols
    
    for row_i in range(n_rows):
        cols = st.columns(n_cols)
        for col_i in range(n_cols):
            idx = row_i * n_cols + col_i
            if idx < len(vars_to_show):
                var = vars_to_show[idx]
                freq = data[var].value_counts(normalize=True).reset_index()
                freq.columns = ['Modalité', 'Proportion']
                freq['Proportion'] *= 100
                
                fig = px.bar(
                    freq, x='Modalité', y='Proportion',
                    title=f"{var}",
                    color='Proportion',
                    color_continuous_scale='Blues',
                    text=freq['Proportion'].round(1).astype(str) + '%'
                )
                fig.update_traces(textposition='outside')
                fig.update_layout(
                    height=280, margin=dict(t=40, b=10, l=10, r=10),
                    showlegend=False, coloraxis_showscale=False,
                    yaxis_title='%', xaxis_title=''
                )
                cols[col_i].plotly_chart(fig, use_container_width=True)
    
    # Distribution du risque
    st.markdown("### Distribution du Risque par Variable")
    var_risk = st.selectbox("Choisir une variable :", acm_vars_selected)
    
    cross = pd.crosstab(data[var_risk], data['Risque'], normalize='index') * 100
    cross = cross.reset_index().melt(id_vars=var_risk, var_name='Risque', value_name='%')
    
    fig = px.bar(cross, x=var_risk, y='%', color='Risque', barmode='stack',
                 color_discrete_map={'Bon':'#2ECC71','Moyen':'#F39C12','Mauvais':'#E74C3C'},
                 title=f"Répartition du Risque par {var_risk}")
    fig.update_layout(height=380, yaxis_title='% clients', xaxis_title=var_risk)
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 : ASSOCIATIONS (CRAMÉR)
# ═════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 🔗 Matrice des Associations — V de Cramér")
    
    st.markdown("""
    <div class="info-box">
    <b>V de Cramér</b> : mesure l'intensité de l'association entre deux variables catégorielles.<br>
    • 0.0–0.1 : très faible &nbsp;&nbsp; • 0.1–0.3 : modérée &nbsp;&nbsp; • 0.3–0.5 : forte &nbsp;&nbsp; • >0.5 : très forte
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.imshow(
            cramer_mat,
            text_auto='.2f',
            color_continuous_scale='RdYlGn',
            zmin=0, zmax=1,
            title="Matrice des V de Cramér",
            aspect='auto'
        )
        fig.update_layout(height=500, coloraxis_colorbar_title="V de Cramér")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🔝 Top associations")
        pairs = []
        vars_list = acm_vars_selected
        for i, c1 in enumerate(vars_list):
            for j, c2 in enumerate(vars_list):
                if i < j:
                    pairs.append({'Var 1': c1, 'Var 2': c2,
                                  'V Cramér': cramer_mat.loc[c1, c2]})
        top = pd.DataFrame(pairs).sort_values('V Cramér', ascending=False).head(8)
        
        for _, row in top.iterrows():
            v = row['V Cramér']
            color = '#2ECC71' if v > 0.3 else ('#F39C12' if v > 0.1 else '#E74C3C')
            bar_width = int(v * 100)
            st.markdown(f"""
            <div style="margin:6px 0; padding:8px; background:white; border-radius:8px; box-shadow:0 1px 4px rgba(0,0,0,0.1);">
                <div style="font-size:12px; color:#666;">{row['Var 1']} ↔ {row['Var 2']}</div>
                <div style="display:flex; align-items:center; gap:8px; margin-top:4px;">
                    <div style="flex:1; height:8px; background:#eee; border-radius:4px;">
                        <div style="width:{bar_width}%; height:8px; background:{color}; border-radius:4px;"></div>
                    </div>
                    <span style="font-weight:bold; color:{color};">{v:.3f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 : VALEURS PROPRES
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## ⚡ Valeurs Propres & Inertie Expliquée")
    
    st.markdown("""
    <div class="info-box">
    Les <b>valeurs propres</b> mesurent l'information capturée par chaque axe factoriel.<br>
    <b>Règle :</b> on retient les axes qui expliquent cumulativement ≥ 70-80% de l'inertie totale.
    </div>
    """, unsafe_allow_html=True)
    
    n_show = min(8, len(eigs))
    axes_labels = [f'Axe {i+1}' for i in range(n_show)]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = go.Figure()
        fig.add_bar(x=axes_labels, y=eigs[:n_show], name='Valeur propre',
                    marker_color='steelblue', marker_line_width=0)
        fig.add_scatter(x=axes_labels, y=eigs[:n_show], mode='lines+markers',
                        line=dict(color='darkblue', width=2),
                        marker=dict(size=8, color='darkblue'))
        kaiser = 1/len(acm_vars_selected)
        fig.add_hline(y=kaiser, line_dash='dash', line_color='red',
                      annotation_text=f"Kaiser (1/K={kaiser:.3f})")
        fig.update_layout(title='Valeurs Propres', height=350,
                          yaxis_title='λ', margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        colors = ['#2196F3' if p > 5 else '#B0BEC5' for p in pct_var[:n_show]]
        fig = go.Figure()
        fig.add_bar(x=axes_labels, y=pct_var[:n_show],
                    marker_color=colors, text=[f'{p:.1f}%' for p in pct_var[:n_show]],
                    textposition='outside')
        fig.update_layout(title='% Variance Expliquée', height=350,
                          yaxis_title='%', margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure()
        fig.add_scatter(x=axes_labels, y=cumul_var[:n_show], mode='lines+markers',
                        line=dict(color='#4CAF50', width=2.5),
                        marker=dict(size=9), fill='tozeroy', fillcolor='rgba(76,175,80,0.1)')
        fig.add_hline(y=80, line_dash='dash', line_color='orange',
                      annotation_text='80%')
        fig.add_hline(y=70, line_dash='dot', line_color='red',
                      annotation_text='70%')
        fig.update_layout(title='Variance Cumulée', height=350,
                          yaxis_title='%', yaxis_range=[0, 110], margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    
    # Tableau récapitulatif
    st.markdown("### Tableau des Valeurs Propres")
    eig_df = pd.DataFrame({
        'Axe': axes_labels,
        'Valeur propre': [f'{e:.4f}' for e in eigs[:n_show]],
        '% Variance': [f'{p:.2f}%' for p in pct_var[:n_show]],
        '% Cumulé': [f'{c:.2f}%' for c in cumul_var[:n_show]],
        'Recommandation': ['✅ Retenir' if cumul_var[i] <= 80 else '⚪ Optionnel'
                           for i in range(n_show)]
    })
    st.dataframe(eig_df, use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 : PLAN FACTORIEL
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## 🗺️ Plan Factoriel")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("### Options")
        show_individuals = st.checkbox("Afficher les individus", True)
        show_modalities = st.checkbox("Afficher les modalités", True)
        color_by = st.selectbox("Colorer individus par :", ['Risque', 'Cluster'] + acm_vars_selected)
        point_size = st.slider("Taille des points individus", 3, 12, 5)
        point_opacity = st.slider("Opacité individus", 0.1, 1.0, 0.4)
        
        st.markdown("---")
        st.markdown(f"""
        **Axe {axe_x}** : {pct_var[axe_x-1]:.1f}% variance  
        **Axe {axe_y}** : {pct_var[axe_y-1]:.1f}% variance  
        **Total plan** : {pct_var[axe_x-1] + pct_var[axe_y-1]:.1f}%
        """)
    
    with col1:
        fig = go.Figure()
        
        # Individus
        if show_individuals and ax_x_col in row_c.columns and ax_y_col in row_c.columns:
            color_vals = data[color_by].astype(str)
            color_map = {}
            if color_by == 'Risque':
                color_map = {'Bon': '#2ECC71', 'Moyen': '#F39C12', 'Mauvais': '#E74C3C'}
            
            for cat in sorted(color_vals.unique()):
                mask = color_vals == cat
                color = color_map.get(cat, None)
                fig.add_trace(go.Scatter(
                    x=row_c.loc[mask, ax_x_col],
                    y=row_c.loc[mask, ax_y_col],
                    mode='markers', name=f'{cat}',
                    legendgroup='individus', legendgrouptitle_text='Individus',
                    marker=dict(size=point_size, opacity=point_opacity,
                                color=color if color else None),
                    hovertemplate=f'{color_by}: {cat}<br>Axe {axe_x}: %{{x:.3f}}<br>Axe {axe_y}: %{{y:.3f}}<extra></extra>'
                ))
        
        # Modalités
        if show_modalities and ax_x_col in col_c.columns and ax_y_col in col_c.columns:
            var_colors_map = {
                v: px.colors.qualitative.Set2[i] for i, v in enumerate(acm_vars_selected)
            }
            
            for var in acm_vars_selected:
                mods_for_var = [m for m in col_c.index 
                                if any(str(val) in str(m) for val in data[var].unique())]
                if not mods_for_var:
                    continue
                
                x_vals = [col_c.loc[m, ax_x_col] for m in mods_for_var]
                y_vals = [col_c.loc[m, ax_y_col] for m in mods_for_var]
                labels = [str(m).split('_', 1)[-1] if '_' in str(m) else str(m) for m in mods_for_var]
                
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='markers+text', name=var,
                    legendgroup='modalites', legendgrouptitle_text='Modalités',
                    text=labels, textposition='top center',
                    marker=dict(size=14, symbol='diamond',
                                color=var_colors_map[var],
                                line=dict(width=2, color='white')),
                    hovertemplate='<b>%{text}</b><br>' + var +
                                  f'<br>Axe {axe_x}: %{{x:.3f}}<br>Axe {axe_y}: %{{y:.3f}}<extra></extra>'
                ))
        
        fig.add_hline(y=0, line_dash='dash', line_color='gray', line_width=1, opacity=0.5)
        fig.add_vline(x=0, line_dash='dash', line_color='gray', line_width=1, opacity=0.5)
        
        fig.update_layout(
            title=dict(text=f'Plan Factoriel ACM — Axes {axe_x} & {axe_y}', font=dict(size=16)),
            xaxis_title=f'Axe {axe_x} ({pct_var[axe_x-1]:.1f}%)',
            yaxis_title=f'Axe {axe_y} ({pct_var[axe_y-1]:.1f}%)',
            height=580, hovermode='closest',
            plot_bgcolor='rgba(250,250,252,0.9)',
            legend=dict(groupclick='toggleitem')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Table des coordonnées des modalités
    with st.expander("📋 Coordonnées des modalités"):
        display_coords = col_c[[ax_x_col, ax_y_col]].copy()
        display_coords.columns = [f'Coordonnée Axe {axe_x}', f'Coordonnée Axe {axe_y}']
        display_coords['Distance à l\'origine'] = np.sqrt(
            col_c[ax_x_col]**2 + col_c[ax_y_col]**2
        ).round(4)
        st.dataframe(display_coords.round(4).sort_values('Distance à l\'origine', ascending=False),
                     use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 : CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(f"## 👥 Segmentation K-Means (K={k_clusters})")
    
    sil_score = silhouette_score(X_cl, data['Cluster'])
    
    col1, col2, col3 = st.columns(3)
    col1.metric(" Nombre de segments", k_clusters)
    col2.metric(" Score Silhouette", f"{sil_score:.3f}")
    col3.metric(" Qualité", "Bonne" if sil_score > 0.4 else ("Moyenne" if sil_score > 0.2 else "Faible"))
    
    st.markdown("---")
    
    # Plan factoriel avec clusters
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.scatter(
            data, x='ACM_Axe1', y='ACM_Axe2',
            color='Cluster', color_discrete_sequence=px.colors.qualitative.Set1,
            hover_data=['Risque', 'Profession', 'Logement'],
            title='Clusters K-Means sur le Plan Factoriel ACM',
            labels={'ACM_Axe1': f'Axe 1 ({pct_var[0]:.1f}%)',
                    'ACM_Axe2': f'Axe 2 ({pct_var[1]:.1f}%)'}
        )
        fig.update_traces(marker=dict(size=5, opacity=0.5))
        
        # Centroïdes
        for c in range(k_clusters):
            mask = data['Cluster'] == c
            cx = data.loc[mask, 'ACM_Axe1'].mean()
            cy = data.loc[mask, 'ACM_Axe2'].mean()
            fig.add_scatter(
                x=[cx], y=[cy], mode='markers+text',
                marker=dict(size=20, symbol='star', color='white',
                           line=dict(width=2, color='black')),
                text=[f'C{c}'], textposition='middle center',
                showlegend=False, name=f'Centroïde {c}'
            )
        
        fig.add_hline(y=0, line_dash='dash', opacity=0.3)
        fig.add_vline(x=0, line_dash='dash', opacity=0.3)
        fig.update_layout(height=480)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Composition des Clusters")
        for c in range(k_clusters):
            sub = data[data['Cluster'] == c]
            n = len(sub)
            pct = n/len(data)*100
            risque_mode = sub['Risque'].value_counts().index[0]
            prof_mode = sub['Profession'].value_counts().index[0]
            
            color = {'Bon':'#E8F5E9','Moyen':'#FFF3E0','Mauvais':'#FFEBEE'}.get(risque_mode,'#EEE')
            border = {'Bon':'#2E7D32','Moyen':'#E65100','Mauvais':'#C62828'}.get(risque_mode,'#999')
            
            st.markdown(f"""
            <div style="background:{color}; border-left:4px solid {border}; 
                        border-radius:8px; padding:12px; margin:8px 0;">
                <b>Cluster {c}</b> — {n} clients ({pct:.0f}%)<br>
                <small>Risque dominant : <b>{risque_mode}</b><br>
                Profession : {prof_mode}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Profil détaillé
    st.markdown("### Profil Détaillé par Cluster")
    
    cluster_choice = st.selectbox("Sélectionner un cluster :", range(k_clusters),
                                   format_func=lambda c: f"Cluster {c}")
    
    sub = data[data['Cluster'] == cluster_choice]
    
    cols = st.columns(len(acm_vars_selected))
    for i, var in enumerate(acm_vars_selected):
        freq = sub[var].value_counts(normalize=True).reset_index()
        freq.columns = ['Modalité', 'Proportion']
        freq['Proportion'] *= 100
        fig = px.pie(freq, names='Modalité', values='Proportion',
                     title=var, hole=0.4)
        fig.update_layout(height=220, margin=dict(t=35, b=5, l=5, r=5),
                          showlegend=False)
        cols[i].plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 : SCORING CLIENT
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## 🔮 Scoring d'un Nouveau Client")
    
    st.markdown("""
    <div class="info-box">
    Renseignez le profil d'un nouveau client pour obtenir son positionnement dans l'espace ACM 
    et sa probabilité de risque estimée.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Profil du Client")
        
        client_data = {}
        var_options = {
            'Marche': ['Positif', 'Positif_fort', 'Nul', 'Débiteur_faible', 'Débiteur_fort', 'Sans_compte'],
            'Apport': ['Elevé', 'Moyen', 'Faible', 'Très_faible'],
            'Impaye': ['Aucun', 'Ponctuel', 'Répété'],
            'Assurance': ['Complète', 'Décès_invalidité', 'Aucune'],
            'Endettement': ['Faible', 'Modéré', 'Elevé', 'Très_élevé'],
            'Famille': ['Marié', 'Célibataire', 'Divorcé'],
            'Enfants': ['0', '1-2', '3+'],
            'Logement': ['Propriétaire', 'Locataire', 'Hébergé'],
            'Profession': ['Cadre', 'Fonctionnaire', 'Employé_qualifié', 'Ouvrier', 'Indépendant', 'Sans_emploi'],
            'Intitule': ['Immobilier', 'Voiture_neuve', 'Voiture_occasion', 'Consommation']
        }
        
        for var in acm_vars_selected:
            if var in var_options:
                options = var_options[var]
                # Options disponibles dans le dataset
                available = sorted(data[var].unique().tolist())
                combined = list(dict.fromkeys(options + available))
                client_data[var] = st.selectbox(f"{var} :", combined, key=f"score_{var}")
        
        age_client = st.slider("Âge :", 20, 75, 35)
    
    with col2:
        st.markdown("### Résultat du Scoring")
        
        if st.button("🔮 Analyser ce client", type="primary", use_container_width=True):
            # Créer un dataframe avec le client
            client_df = pd.DataFrame([client_data])[acm_vars_selected]
            
            # Transformer avec le modèle ACM
            try:
                client_coords = mca_model.transform(client_df)
                client_coords.columns = [f'Axe_{i+1}' for i in range(client_coords.shape[1])]
                
                cx1 = client_coords['Axe_1'].values[0]
                cx2 = client_coords['Axe_2'].values[0] if 'Axe_2' in client_coords.columns else 0
                
                # Cluster le plus proche
                if 'Axe_3' in client_coords.columns:
                    client_vec = client_coords.iloc[0, :3].values.reshape(1, -1)
                else:
                    client_vec = client_coords.iloc[0, :min(3,client_coords.shape[1])].values.reshape(1, -1)
                
                # Padding si nécessaire
                if client_vec.shape[1] < X_cl.shape[1]:
                    pad = np.zeros((1, X_cl.shape[1] - client_vec.shape[1]))
                    client_vec = np.hstack([client_vec, pad])
                
                cluster_pred = km.predict(client_vec)[0]
                
                # Profil du cluster
                sub_cluster = data[data['Cluster'] == cluster_pred]
                risque_mode = sub_cluster['Risque'].value_counts(normalize=True)
                
                pct_bon = risque_mode.get('Bon', 0) * 100
                pct_moyen = risque_mode.get('Moyen', 0) * 100
                pct_mauvais = risque_mode.get('Mauvais', 0) * 100
                
                # Affichage
                risque_dominant = risque_mode.index[0]
                color_map = {'Bon': '🟢', 'Moyen': '🟡', 'Mauvais': '🔴'}
                
                st.markdown(f"""
                <div style="background:white; border-radius:12px; padding:20px; 
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin:10px 0;">
                    <h3 style="color:#1A237E; margin:0 0 15px 0;">Résultat de l'Analyse</h3>
                    <div style="font-size:2rem; margin:10px 0;">{color_map[risque_dominant]} Risque {risque_dominant}</div>
                    <div style="color:#666; margin:5px 0;">Affecté au Cluster {cluster_pred}</div>
                    <hr style="border-top:1px solid #eee; margin:15px 0;">
                    <div style="font-size:13px; color:#555; margin-bottom:5px;"><b>Probabilités estimées :</b></div>
                    <div style="margin:4px 0;">🟢 Bon : <b>{pct_bon:.0f}%</b></div>
                    <div style="margin:4px 0;">🟡 Moyen : <b>{pct_moyen:.0f}%</b></div>
                    <div style="margin:4px 0;">🔴 Mauvais : <b>{pct_mauvais:.0f}%</b></div>
                    <hr style="border-top:1px solid #eee; margin:15px 0;">
                    <div style="font-size:12px; color:#888;">
                    Coordonnées ACM : Axe1={cx1:.3f}, Axe2={cx2:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Positionnement sur le plan factoriel
                fig = go.Figure()
                
                # Fond : tous les clients
                for risque_val, color in [('Bon','#2ECC71'),('Moyen','#F39C12'),('Mauvais','#E74C3C')]:
                    mask = data['Risque'] == risque_val
                    fig.add_scatter(
                        x=data.loc[mask, 'ACM_Axe1'], y=data.loc[mask, 'ACM_Axe2'],
                        mode='markers', name=risque_val,
                        marker=dict(color=color, size=4, opacity=0.25),
                        hoverinfo='skip'
                    )
                
                # Point client
                fig.add_scatter(
                    x=[cx1], y=[cx2],
                    mode='markers+text', name='Nouveau client',
                    text=['⭐ Vous'], textposition='top center',
                    textfont=dict(size=13, color='black', family='Arial Black'),
                    marker=dict(size=22, symbol='star', color='gold',
                               line=dict(width=3, color='black'))
                )
                
                fig.add_hline(y=0, line_dash='dash', opacity=0.3)
                fig.add_vline(x=0, line_dash='dash', opacity=0.3)
                fig.update_layout(
                    title='Positionnement sur le Plan Factoriel',
                    xaxis_title=f'Axe 1 ({pct_var[0]:.1f}%)',
                    yaxis_title=f'Axe 2 ({pct_var[1]:.1f}%)',
                    height=380
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Certaines modalités saisies ne sont pas dans le modèle. Ajustez les valeurs. ({e})")
        
        else:
            st.info("👆 Renseignez le profil du client et cliquez sur 'Analyser'")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#999; font-size:12px; padding:10px 0;">
    🏦 ACM Crédit Bancaire — Projet Portfolio &nbsp;|&nbsp; 
    Python · prince · sklearn · plotly · streamlit &nbsp;|&nbsp;
    Analyse des Correspondances Multiples
</div>
""", unsafe_allow_html=True)
