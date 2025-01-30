# **Automated Intercom System - Voice Recognition using Deep Learning**

## **Project Overview**

This project aims to develop an **automated intercom system** that recognizes whether a speaker belongs to an allowed group (Class 1) or not (Class 0). The system leverages **deep learning models trained on spectrograms** of audio data, making it robust for real-world deployment.

## **Dataset**

We use the **DAPS (Device and Produced Speech) Dataset** ([Zenodo Link](https://zenodo.org/records/4660670)), which contains **1800 audio files** of varying length and quality.

- **Class 1 (Allowed Users)**: F1, F7, F8, M3, M6, M8 (6 speakers)
- **Class 0 (Unknown Users)**: All other 14 speakers

This results in a **binary classification** problem.

## **User Experience (UX/UI)**

The final system is implemented as a simple **UI** app that allows users to interact with the model:

1. The user initializes the listening procedure.
2. The system captures audio using the laptop microphone.
3. The model predicts the speaker’s class and displays the result.
4. The user uploads a prerecorded audio or uploaded one and sees the result in the app window. 

Additionally, the UX/UI functionality includes **looping the audio after trimming silence** to ensure a minimum duration of **3 seconds** before processing.

## **Modeling Approach**

We experiment with several models and optimize for the **F1-score**, the chosen evaluation metric. The best-performing models are:

### **Deep Learning Models:**

- **Simplified ResNet** *(Best Model: F1-score: 0.98 Validation, 0.9760 Test)*
- **CNN (Adam + LR Scheduler)**
- **Monte Carlo Dropout (MC Dropout) for Uncertainty Estimation**
- **MLP (Baseline for comparison)**

### **Classical Machine Learning Models:**

- **Random Forest Classifier** with **RandomizedSearchCV** hyperparameter tuning.

## **Preprocessing Steps**

**Why Preprocessing?** Raw audio files contain excess information like silence, noise, and irrelevant frequency components. To improve classification accuracy, we preprocess the data before converting it into spectrograms.

### **Key Preprocessing Techniques Applied:**

1. **Trimming Silence**: Removing long silences ensures only meaningful speech is retained.
2. **Noise Reduction**: Background noise is filtered to enhance clarity.
3. **Segmenting Audio**: Longer recordings are cut into **3-second** chunks for consistency.
4. **Feature Extraction**: Convert audio into **Mel-spectrograms** (64 mel bands, 16kHz sampling rate).

## **Exploratory Data Analysis (EDA)**

### **Spectrogram Analysis**

- **Raw Spectrograms (RDS)**:
  - Can be over **2 minutes long**.
  - Contain **excess noise and silence**, leading to unclear patterns.
  - Show **denser frequency ranges**, likely due to background noise.
- **Processed Spectrograms**:
  - **Much shorter** (3 sec max), focusing on essential speech components.
  - **Clearer patterns**, thanks to effective noise reduction.
  - **More distinct frequency features**, improving model training.

### **Histograms: Mean & Variance in Intensity**

We analyzed the distribution of **mean intensity and variance** before and after preprocessing:

- **Raw Spectrograms:**
  - Lower mean intensity due to background noise.
  - Higher variance in intensity, suggesting uneven recording conditions.
- **Processed Spectrograms:**
  - Higher mean intensity after **normalization**.
  - Lower variance due to **smoother, more consistent** features.

These improvements ensure the model focuses on essential **voice patterns**, rather than noise and silence.

## **Monte Carlo Dropout for Uncertainty Estimation**

To estimate classification confidence, **Monte Carlo Dropout (MC Dropout)** was implemented with **T=1000 stochastic forward passes**, which significantly improved precision-recall balance. The final F1-score for **MC Dropout: 0.9030**. MC Dropout exhibits higher uncertainty compared to the ensemble method, as expected, due to its stochastic nature.
The ensemble model has lower uncertainty and a tighter spread, thus more stable predictions.

Additionally, an **ensemble of 5 CNN models** was trained for comparison, achieving an F1-score of **0.9307**.

## **Model Pruning and Optimization**

To improve model efficiency, we implemented **Layer-Wise Pruning**, inspired by the **Lottery Ticket Hypothesis**.

1. **Original Model:** ResNet with **100% parameters** (F1-score: 0.9760).
2. **Pruned Model:** ResNet with **50% sparsity** (Layer-Wise Pruning).
   - **Fine-tuned F1-score: 0.9683** → **Minimal quality loss!**
   - **Inference time improved** from **57.49 ms → 43.74 ms per batch.**
3. **Reinitialized Pruned Model:** Resetting pruned weights to their **original initialization** and training again.
   - **Inference time: 43.74 ms** (faster than original model).

## **Performance Comparison**

| Model                           | F1-score (Test) | Inference Time (ms/batch) |
| ------------------------------- | --------------- | ------------------------- |
| **Original ResNet**             | 0.9760          | 57.49                     |
| **Fine-Tuned Pruned ResNet**    | 0.9683          | 48.77                     |
| **Reinitialized Pruned ResNet** | 0.9683          | 43.74                     |
| **MC Dropout**                  | 0.9030          | -                         |
| **Ensemble Model**              | 0.9307          | -                         |

## **How to Run the Project**

### **Running the UI Application**

To run the **front-end GUI**, execute:

```bash
pyinstaller front.spec
```
Run the packed executable. 

### **Running the Model Pipeline**

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the preprocessing pipeline**:
   ```python
   from ml_project_laris import process_audio_to_spectrograms_and_save_in_chunks
   process_audio_to_spectrograms_and_save_in_chunks('./daps', sr=16000, n_mels=64, cut_length=3, save_dir='./npy_spectrograms')
   ```

Alternatively, you can run the Jupyter Notebooks to analyze the dataset, visualize results, or run the whole pipeline of preprocessing, training and predicting.
🚀 **For detailed reports and explanations, go to** `FinalReport.ipynb` **and** `ml_project_code.ipynb`. 🚀

