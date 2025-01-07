# SMS Spam Detection

## 📜 Project Overview
This project is a machine learning-based solution for detecting spam messages in SMS. It classifies messages as **spam** or **ham** (not spam) using models like **Naive Bayes** and **K-Nearest Neighbors (KNN)**.

## 📂 Project Structure
- `data/spam_ham_dataset.csv` - The dataset used for training and testing the models.
- `notebooks/SMS_spam_detection.ipynb` - The Jupyter Notebook containing the code for:
  - Data preprocessing
  - Model training
  - Model evaluation
- `requirements.txt` - The Python dependencies required to run the project.

## 🚀 Features
- Preprocessing of SMS messages (e.g., removing special characters, converting to lowercase, etc.).
- Implementing machine learning models (Naive Bayes and KNN) using `scikit-learn`.
- Evaluation of model performance using metrics like accuracy, precision, and recall.
- Visualization of results with plots.

## 📊 Dataset
The dataset is sourced from [Kaggle](https://www.kaggle.com/), containing labeled SMS messages categorized as **spam** or **ham**.

## 🔧 How to Run
Follow these steps to run the project locally:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/JanaviN7/SMS-spam-detection.git
   cd SMS-spam-detection
   ```

2. **Install the required libraries**:
   Use the following command to install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook notebooks/SMS_spam_detection.ipynb
   ```

4. **Run the Notebook**:
   Execute all cells in sequence to preprocess the data, train the models, and view the evaluation results.

## 📚 Dependencies
The project requires the following Python libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`

These can be installed using the `requirements.txt` file.

## 🔍 Evaluation Metrics
The project evaluates the models using:
- Accuracy
- Precision
- Recall
- Confusion Matrix

## 🤝 Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## 📄 License
This project is licensed under the MIT License.

