# Emoji Prediction using Twitter Data

A comprehensive machine learning project that predicts emoji labels from Twitter text data using three different approaches: Support Vector Machine (SVM), Bidirectional LSTM, and a hybrid CNN + Bidirectional LSTM model.

## 📊 Project Overview

This project tackles the challenge of predicting appropriate emojis for Twitter posts using natural language processing and deep learning techniques. The models are trained on a dataset of 50,000 tweets labeled with 20 different emoji categories.

## 📁 Dataset

The project uses the following data files located in the `Data/` directory:
- `us_train.text` - Contains 50,000 tweets for training
- `us_train.labels` - Contains corresponding emoji labels (0-19)  
- `us_mapping.txt` - Maps emoji labels to their Unicode representations and meanings

### Emoji Categories (20 classes):
| Label | Emoji | Meaning |
|-------|-------|---------|
| 0 | ❤️ | Red heart |
| 1 | 😍 | Smiling face with heart-eyes |
| 2 | 😂 | Face with tears of joy |
| 3 | 💕 | Two hearts |
| 4 | 🔥 | Fire |
| 5 | 😊 | Smiling face with smiling eyes |
| 6 | 😎 | Smiling face with sunglasses |
| 7 | ✨ | Sparkles |
| 8 | 💙 | Blue heart |
| 9 | 😘 | Face blowing a kiss |
| 10 | 📷 | Camera |
| 11 | 🇺🇸 | United States |
| 12 | ☀️ | Sun |
| 13 | 💜 | Purple heart |
| 14 | 😉 | Winking face |
| 15 | 💯 | Hundred points |
| 16 | 😁 | Beaming face with smiling eyes |
| 17 | 🎄 | Christmas tree |
| 18 | 📸 | Camera with flash |
| 19 | 😜 | Winking face with tongue |

## 🔧 Data Preprocessing

The preprocessing pipeline includes:

1. **Text Normalization**:
   - Convert to lowercase
   - Remove HTML tags
   - Remove URLs
   - Remove punctuation
   - Remove @mentions
   - Remove numbers

2. **NLP Processing**:
   - Remove English stopwords
   - Apply lemmatization using WordNet

3. **Data Handling**:
   - Remove duplicate tweets (116 duplicates found)
   - Handle class imbalance using Random Over Sampling
   - Final dataset: 214,560 balanced samples

4. **Feature Engineering**:
   - Sentiment analysis using VADER (found low correlation with labels)
   - Tweet length analysis (found low correlation with labels)

## 🤖 Models Implemented

### 1. Support Vector Machine (SVM)
**File**: `Emoji_detection_ML_SVM.ipynb`

**Architecture**:
- Feature extraction using Count Vectorization (Bag of Words)
- 55,352 unique features
- SVM classifier with default RBF kernel

**Performance**:
- **Accuracy**: 92.20%
- **Macro Average F1-Score**: 0.92
- **Training time**: Fast
- **Best for**: Traditional ML approach with good interpretability

### 2. Bidirectional LSTM
**File**: `Emoji_detection_DL_Bidirectional_LSTMs.ipynb`

**Architecture**:
```
- Embedding Layer (40 dimensions)
- Bidirectional LSTM (80 units, return_sequences=True)
- Dropout (0.5)
- Bidirectional LSTM (40 units)
- Dropout (0.5)
- Dense Layer (20 units, softmax activation)
```

**Performance**:
- **Accuracy**: 92.17%
- **Macro Average F1-Score**: 0.92
- **Training**: Early stopping after 9 epochs
- **Best for**: Capturing sequential patterns in tweet text

### 3. CNN + Bidirectional LSTM (Hybrid)
**File**: `Emoji_detection_CNN_Bidirectional_LSTMs.ipynb`

**Architecture**:
```
- Embedding Layer (50 dimensions)
- 1D Convolutional Layer (64 filters, kernel_size=5)
- MaxPooling1D (pool_size=2)
- Bidirectional LSTM (80 units, return_sequences=True)
- Dropout (0.5)
- Bidirectional LSTM (40 units)
- Dropout (0.5)
- Dense Layer (20 units, softmax activation)
```

**Performance**:
- **Accuracy**: 92.67% ⭐ **(Best Performance)**
- **Macro Average F1-Score**: 0.92
- **Training**: Early stopping after 7 epochs
- **Best for**: Combining local feature extraction (CNN) with sequential modeling (LSTM)

## 📈 Model Comparison

| Model | Accuracy | F1-Score | Parameters | Training Time | Strengths |
|-------|----------|----------|------------|---------------|-----------|
| SVM | 92.20% | 0.92 | N/A | Fast | Interpretable, robust |
| Bidirectional LSTM | 92.17% | 0.92 | 2.5M | Medium | Sequential modeling |
| **CNN + Bi-LSTM** | **92.67%** | **0.92** | **3.2M** | **Medium** | **Best accuracy, hybrid approach** |

## 🛠️ Requirements

```bash
# Core libraries
numpy
pandas
matplotlib
seaborn
scikit-learn

# NLP libraries
nltk
textblob
vaderSentiment

# Deep Learning
tensorflow
keras

# Data handling
imbalanced-learn
```

## 🚀 Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KartikGarg20526/Emoji-Prediction-using-twitter-data.git
   cd Emoji-Prediction-using-twitter-data
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the models**:
   - For SVM: Open `Emoji_detection_ML_SVM.ipynb`
   - For Bidirectional LSTM: Open `Emoji_detection_DL_Bidirectional_LSTMs.ipynb`
   - For CNN + Bi-LSTM: Open `Emoji_detection_CNN_Bidirectional_LSTMs.ipynb`

## 📝 Key Findings

1. **Class Imbalance**: Original dataset was heavily imbalanced (Label 0: 10,728 vs Label 19: 1,213)
2. **Feature Correlation**: Tweet length and sentiment scores showed low correlation with emoji labels
3. **Model Performance**: All three models achieved similar accuracy (~92%), with the hybrid CNN + Bi-LSTM slightly outperforming
4. **Overfitting Prevention**: Early stopping and dropout layers were crucial for generalization
5. **Preprocessing Impact**: Comprehensive text preprocessing significantly improved model performance

## 🔍 Model Analysis

### Performance by Emoji Category:
- **High Performance** (F1 > 0.95): Fire 🔥, Camera 📷, Sun ☀️, Hundred points 💯
- **Medium Performance** (F1 = 0.85-0.95): Most emoji categories
- **Lower Performance** (F1 < 0.85): Red heart ❤️ (due to class overlap and ambiguity)

### Training Insights:
- **Early Stopping**: Prevented overfitting in deep learning models
- **Bidirectional Processing**: Improved context understanding compared to unidirectional LSTM
- **CNN Features**: Local pattern extraction complemented sequential LSTM processing

## 📊 Evaluation Metrics

Each model includes comprehensive evaluation:
- **Confusion Matrix**: Detailed class-wise performance visualization
- **Classification Report**: Precision, recall, and F1-score for each emoji class
- **Training History**: Loss and accuracy curves for deep learning models
- **Cross-validation**: Robust performance estimation

## 🎯 Future Improvements

1. **Advanced Architectures**: Transformer-based models (BERT, RoBERTa)
2. **Feature Enhancement**: Include user metadata, tweet context
3. **Multi-label Classification**: Support multiple emoji predictions
4. **Real-time Deployment**: Web service for live emoji prediction
5. **Expanded Dataset**: Include more recent tweets and emoji categories

## 👨‍💻 Author

**Kartik Garg**
- GitHub: [KartikGarg20526](https://github.com/KartikGarg20526)
- Project: Emoji Prediction using Twitter Data

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*This project demonstrates the application of both traditional machine learning and deep learning approaches to natural language processing tasks, showcasing the effectiveness of different modeling strategies for emoji prediction from social media text.*