# Student Academic Performance Segmentation

An unsupervised machine-learning project that explores student cohorts using study hours, attendance rate, assignments submitted, and exam scores.

## Objectives
- Extract and inspect academic-behavior features.
- Apply feature scaling before distance-based clustering.
- Select candidate cluster counts using the Elbow Method and Silhouette Coefficient.
- Compare K-Means and Ward Hierarchical Clustering.
- Visualize clusters with PCA (2D/3D) and optional t-SNE.
- Summarize cluster characteristics as supportive student personas.

## Repository structure
```
student-academic-performance-segmentation/
├── data/student_performance.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── clustering.py
│   └── visualization.py
├── student_segmentation.ipynb
├── student_personas.md
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup
Python 3.10+ is recommended.

```bash
git clone https://github.com/KageFlow/student-academic-performance-segmentation.git
cd student-academic-performance-segmentation
python -m venv .venv
```

Activate the environment:
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Install packages:
```bash
pip install -r requirements.txt
```

## Run the notebook
Open `student_segmentation.ipynb` in Jupyter or Google Colab and run cells in order from the repository root. The notebook imports helper functions from `src/`, so keep the folder structure intact.

## Dataset schema
| Column | Meaning |
|---|---|
| student_id | Demo identifier |
| study_hours | Approximate study hours per day |
| attendance_rate | Attendance percentage |
| assignments_submitted | Assignments submitted (demo scale: 0–10) |
| exam_score | Exam score (0–100) |

The included CSV is **synthetic demonstration data**, not real student records. It is designed to make the pipeline runnable and illustrate cohort patterns. For any real analysis, use appropriately authorized and anonymized data, document its source and limitations, and follow applicable privacy rules.

## Methodology and mathematical justification
### Feature scaling
The project uses StandardScaler:
`z = (x - mean) / standard_deviation`.
Without scaling, a feature with a larger numeric range can dominate Euclidean distances. Standardization places features on comparable scales, although it does not guarantee that every feature is equally important.

### K-Means
K-Means assigns each observation to the nearest centroid and iteratively minimizes within-cluster sum of squares:
`inertia = sum over clusters of sum over points of squared distance to centroid`.
The Elbow Method examines how inertia decreases as k increases. A bend suggests diminishing returns, but can be subjective.

### Silhouette Coefficient
For each sample, silhouette compares mean intra-cluster distance (a) with the nearest-cluster mean distance (b):
`s = (b - a) / max(a, b)`.
Scores range from -1 to 1; higher values generally indicate better-separated clusters. It should be interpreted alongside cluster size, stability, and usefulness.

### Hierarchical clustering
Ward linkage merges clusters to minimize the increase in within-cluster variance. It provides a second clustering approach for comparison. Here, both methods use the same standardized feature matrix and candidate k.

### PCA and t-SNE
PCA projects scaled data onto directions of maximum variance for 2D/3D visualization. t-SNE is an optional nonlinear visualization technique. Neither projection is used as the input to clustering; the clustering is fit on all original scaled features.

## Responsible interpretation and limitations
- Personas describe group averages, not individual ability or potential.
- Clustering is unsupervised; there are no ground-truth labels proving a cluster is “good” or “poor.”
- Synthetic data cannot support real educational conclusions.
- Results depend on selected features, scaling, distance metric, random initialization, and k.
- Use cluster insights to offer supportive resources, never as the sole basis for high-stakes decisions about students.
- Real student records require authorization, anonymization, access controls, and careful handling.

## Technologies
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Jupyter Notebook.
