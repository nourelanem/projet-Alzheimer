# Alzheimer MRI Computer Vision Project

Systeme d'aide a l'analyse de la maladie d'Alzheimer a partir d'IRM cerebrales.

## Workflow valide etape par etape

1. Data Understanding
2. Visual Understanding medical
3. Preprocessing medical
4. Analyse contours, textures et espaces
5. Feature Engineering
6. Data Augmentation medicale
7. Deep Learning
8. Explainability
9. API FastAPI
10. Application Web HTML/CSS/JavaScript + Dashboard
11. Rapport + Demo

## Structure

```text
notebooks/
  01_data_understanding_alzheimer.ipynb
  02_preprocessing_balanced_splits.ipynb
  03_cnn_balanced_training_prediction.ipynb
  04_gradcam_explainability.ipynb
src/
  preprocessing.py
  visualization.py
  features.py
  augmentation.py
api/
  main.py
app/
  index.html
  app.js
  styles.css
reports/
data/alzheimer/
  train/
  test/
```

## Regle de validation

Chaque etape est realisee, testee, puis validee avant de passer a l'etape suivante.
