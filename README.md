# Insurance Fraud Detection System (Generative + Ensemble Models) 🕵️‍♀️

This project demonstrates an end-to-end **fraud detection** pipeline:

- Synthetic insurance claims dataset
- Baseline ensemble model (RandomForest)
- GAN-based synthetic fraud sample generator (for experimentation)
- REST API to score claims

---

## 🧱 Components

- `data/claims.csv` – synthetic claims dataset  
- `eda/explore.py` – quick EDA script  
- `model/train_ensemble.py` – trains a RandomForest classifier  
- `model/gan_synthetic.py` – simple GAN to generate synthetic fraud-like data  
- `deployment/api.py` – FastAPI service exposing `/predict_fraud`  

---

## 🛠 Tech Stack

- Python
- scikit-learn
- PyTorch (GAN)
- FastAPI
- Uvicorn
- Pandas / NumPy

---

## 🚀 Setup

```bash
pip install -r requirements.txt
