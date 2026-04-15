# **ACM CRÉDIT BANCAIRE** - Analyse des Correspondances Multiples pour l'Évaluation du Risque

> **Projet Data Science Portfolio** - Application complète d'Analyse des Correspondances Multiples (ACM) avec interface interactive Streamlit

---

## **Contexte et Objectifs**

### **Problématique Métier**
Les institutions bancaires font face quotidiennement au défi de l'évaluation du risque crédit. Comment identifier rapidement les profils clients à risque tout en optimisant le processus de décision ?

### **Solution Proposée**
Ce projet utilise l'**Analyse des Correspondances Multiples (ACM)** pour :
- **Explorer** les profils clients à travers leurs caractéristiques socio-économiques
- **Identifier** les associations cachées entre variables catégorielles
- **Segmenter** les clients en groupes homogènes et interprétables
- **Déployer** un outil interactif pour le scoring en temps réel

---

## **Dataset et Variables**

### **Source des Données**
Dataset simulé de **1000 demandeurs de crédit** avec 10 variables catégorielles et 1 variable quantitative (âge).

### **Variables Analytiques**
| Variable | Description | Modalités | Impact Risque |
|----------|-------------|-----------|---------------|
| `Marche` | Comportement compte courant | Positif, Nul, Débiteur_faible, Débiteur_fort | **Élevé** |
| `Apport` | Niveau d'apport personnel | Élevé, Moyen, Faible, Très_faible | **Moyen** |
| `Impaye` | Historique d'impayés | Aucun, Ponctuel, Répété | **Élevé** |
| `Assurance` | Type d'assurance souscrite | Décès_invalidité, Complète, Aucune | **Moyen** |
| `Endettement` | Niveau d'endettement actuel | Faible, Modéré, Élevé, Très_élevé | **Élevé** |
| `Famille` | Situation familiale | Marié, Célibataire, Divorcé | **Faible** |
| `Enfants` | Nombre d'enfants | 0, 1-2, 3+ | **Faible** |
| `Logement` | Type de logement | Propriétaire, Locataire, Hébergé | **Moyen** |
| `Profession` | Catégorie professionnelle | Cadre, Employé_qualifié, Ouvrier, Sans_emploi | **Élevé** |
| `Intitule` | Type de crédit demandé | Immobilier, Voiture_neuve, Consommation, Voiture_occasion | **Moyen** |
| `Age` | Âge du demandeur | 20-75 ans (quantitatif) | **Faible** |

---

## **Méthodologie Analytique**

### **1. Phase Exploratoire**
```python
# Analyse univariée et bivariée
- Distribution des 10 variables catégorielles  
- Tests du Chi² pour l'indépendance
- Matrice des V de Cramér pour mesurer les associations
```

### **2. Analyse des Correspondances Multiples (ACM)**
```python
# Pipeline ACM avec prince
mca = prince.MCA(
    n_components=8,
    n_iter=10,
    random_state=42,
    engine='sklearn'
)
```

**Résultats ACM :**
- **Valeurs propres** : 8 axes factoriels calculés
- **Variance expliquée** : Axes 1-2 capturent ~45% de l'inertie totale
- **Qualité de représentation** : Cosinus² > 0.5 pour 78% des modalités
- **Contributions** : Identification des variables les plus influentes

### **3. Clustering sur Axes ACM**
```python
# K-Means sur coordonnées factorielles
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
clusters = kmeans.fit_predict(row_coords[:, :3])
```

**Méthodes de validation :**
- **Méthode du coude** : Détection du nombre optimal de clusters
- **Score de Silhouette** : Mesure de la cohésion intra-cluster (0.42)
- **Profiling métier** : Interprétation des 3 segments identifiés

---

## **Résultats et Insights**

### **Axes Factoriels Interprétables**

#### **Axe 1 (28% variance) - Axe du Risque Crédit**
- **Pôle positif** : Compte positif + Aucun impayé + Assurance complète
- **Pôle négatif** : Compte débiteur + Impayés répétés + Sans emploi

#### **Axe 2 (17% variance) - Axe Socio-Professionnel**  
- **Pôle supérieur** : Cadres + Crédit immobilier + Propriétaire
- **Pôle inférieur** : Ouvriers + Crédit consommation + Locataire

### **Segments Clients Identifiés**

| Segment | % Portefeuille | Profil Type | Risque |
|---------|---------------|-------------|--------|
| **Segment 1 - "Stables"** | 40% | Cadres, compte positif, assurance complète | **Faible** |
| **Segment 2 - "Prudents"** | 35% | Employés, profil mixte, risque modéré | **Moyen** |
| **Segment 3 - "Vulnérables"** | 25% | Sans emploi, compte débiteur, impayés | **Élevé** |

---

## **Interface Interactive Streamlit**

### **Architecture de l'Application**
```
app_acm_credit.py
    |
    |-- generate_dataset()     # Génération données simulées
    |-- run_mca()             # Calcul ACM avec cache
    |-- compute_cramer()       # Matrice associations
    |-- clustering_kmeans()    # Segmentation
    |
    |-- 6 onglets thématiques
```

### **Fonctionnalités Détaillées**

#### **1. Vue d'Ensemble**
- KPIs clés : 1000 clients, 10 variables, 3 clusters
- Distributions interactives par variable
- Répartition du risque par segment

#### **2. Matrice des Associations**
- Heatmap interactive des V de Cramér
- Identification visuelle des corrélations fortes
- Tests de significativité (p-values)

#### **3. Analyse Factorielle**
- Éboulis des valeurs propres avec seuils
- Variance expliquée et cumulée
- Aide au choix du nombre d'axes

#### **4. Plans Factoriels**
- Carte des modalités avec légendes par variable
- Projection des individus colorés par risque
- Navigation entre différents plans (1-2, 1-3, 2-3)

#### **5. Clustering**
- Visualisation des 3 segments
- Profils détaillés par cluster
- Métriques de performance (Silhouette, Inertie)

#### **6. Scoring Client**
- Formulaire interactif pour nouveau client
- Projection en temps réel dans l'espace ACM
- Attribution au segment le plus proche
- Évaluation du risque avec recommandations

---

## **Installation et Déploiement**

### **Prérequis**
- Python 3.11+
- Environnement virtuel recommandé

### **Installation Rapide**
```bash
# Clôner et configurer
git clone <repository>
cd ACM_Credit_Analysis

# Créer environnement virtuel
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac

# Installer dépendances
pip install -r requirements.txt
```

### **Fichier requirements.txt**
```txt
pandas>=2.0
numpy>=1.24
prince>=0.12
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15
streamlit>=1.28
scipy>=1.11
```

### **Lancement**
```bash
# Application Streamlit
streamlit run app_acm_credit.py
# Access: http://localhost:8501

# Notebook Jupyter (analyse complète)
jupyter notebook ACM_Credit_Risk_Analysis.ipynb
```

---

## **Stack Technique**

### **Core Technologies**
```
Python 3.11
    |
    |-- Data Processing : pandas, numpy
    |-- Statistical Analysis : scipy, prince
    |-- Machine Learning : scikit-learn
    |-- Visualization : matplotlib, seaborn, plotly
    |-- Web Interface : streamlit
```

### **Architecture Modulaire**
- **Séparation des responsabilités** : Fonctions dédiées par tâche
- **Cache intelligent** : `@st.cache_data` pour optimiser les performances
- **Code réutilisable** : Fonctions exportables pour d'autres projets

---

## **Applications Métier**

### **Cas d'Usage Bancaire**
1. **Pré-qualification** : Filtrage rapide des demandes
2. **Politique tarifaire** : Tarification adaptée au risque
3. **Marketing ciblé** : Offres commerciales par segment
4. **Conformité réglementaire** : Justification des décisions

### **Extensions Possibles**
- **Machine Learning supervisé** : Prédiction du défaut
- **Time-series analysis** : Évolution temporelle du risque
- **API REST** : Intégration avec systèmes existants
- **Dashboard entreprise** : Monitoring en temps réel

---

## **Contributions et Améliorations**

### **Limites Actuelles**
- Dataset simulé (pas de données réelles)
- Modèle statique (pas de mise à jour automatique)
- Interface en français uniquement

### **Pistes d'Amélioration**
- [ ] Intégration de données bancaires réelles
- [ ] Modèle de Machine Learning supervisé
- [ ] API pour intégration externe
- [ ] Multi-langue et internationalisation
- [ ] Tests unitaires et CI/CD

---

## **Auteur et Contact**

**Projet réalisé dans le cadre d'un portfolio Data Science**

*Technologies : Python · ACM · Streamlit · Machine Learning*  
*Domaine : Finance · Risk Management · Data Visualization*

---

*Pour toute question ou collaboration, n'hésitez pas à contacter l'auteur.*
