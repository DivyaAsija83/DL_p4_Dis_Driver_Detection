# DL_p4_Dis_Driver_Detection
Distracted Driver MultiAction Classification

# 🚗 Project: Distracted Driver MultiAction Classification

This project involves building a deep learning model to classify driver behavior from images into one of 10 predefined categories. The goal is to identify distracted driver activities using image classification techniques.

---

## 📂 Dataset

The dataset is provided by the **State Farm Distracted Driver Detection** competition. It contains images of drivers categorized into 10 different classes based on their activities.

🔗 **Download the dataset**:  
[Click here to download (Dropbox link)](https://www.dropbox.com/s/0vyzjcqsdl6cqi2/state-farm-distracted-driver-detection.zip?dl=0)

---

## 🧾 Class Labels

Each image in the dataset belongs to one of the following classes:

| Label | Description                   |
|-------|-------------------------------|
| c0    | Safe driving                  |
| c1    | Texting - right               |
| c2    | Talking on the phone - right |
| c3    | Texting - left                |
| c4    | Talking on the phone - left  |
| c5    | Operating the radio           |
| c6    | Drinking                      |
| c7    | Reaching behind               |
| c8    | Hair and makeup               |
| c9    | Talking to passenger          |

---

## 🛠️ Project Structure

- `notebooks/` — Jupyter/Colab notebooks for training, evaluation, and visualization  
- `src/` — Python scripts for data preprocessing, model building, training pipeline  
- `models/` — Saved model files and checkpoints  
- `data/` — Contains training and validation sets (created after splitting)  
- `README.md` — This documentation  
- `requirements.txt` — Python dependencies  

---

## 🔄 Data Preparation

To prepare the dataset:

1. **Download and extract the dataset**.
2. **Split the data into training and validation sets**:
   - Ensure **class distribution is preserved** across both sets using stratified sampling.
   - You may use tools like `train_test_split` from `sklearn.model_selection` with `stratify` parameter.

---

## 📊 Model Objectives

- Apply **Convolutional Neural Networks (CNN)** for multi-class image classification.
- Use **transfer learning** (e.g., MobileNetV2, EfficientNet, etc.) for better performance.
- Evaluate the model using **accuracy**, **AUC**, **confusion matrix**, and **class-wise performance**.
- Include **visualizations** for loss, accuracy over epochs, and misclassified images.

---

## ✅ Evaluation Metrics

- Overall Accuracy
- Per-class AUC
- Confusion Matrix
- Precision / Recall / F1-score

---

## 🧪 Future Improvements

- Hyperparameter tuning
- Use of advanced architectures (EfficientNet, Vision Transformers)
- Model ensembling
- Real-time inference and deployment via Flask or Streamlit

---

## 📌 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

**Recommended libraries:**

- TensorFlow / Keras
- NumPy / Pandas
- Scikit-learn
- OpenCV / Matplotlib
- seaborn / tqdm

---

## ✍️ Author

Divya Asija
Feel free to connect for feedback, suggestions, or collaborations.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
