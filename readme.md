

# Tweet Classification using Linear SVM & TF-IDF

This project is a machine learning pipeline for binary tweet classification. The primary goal is to process raw tweet data, train a Support Vector Machine (SVM) model, and predict the corresponding labels for a given test set.

The entire workflow is contained within the `nlp_internship.ipynb` Jupyter Notebook and uses the Scikit-learn library for model building and evaluation.

## 📋 Project Workflow

The notebook follows these key steps:

1.  **Data Loading**: The training (`train.csv`) and testing (`test.csv`) datasets are loaded using the pandas library.
2.  **Text Preprocessing**: A custom function, `clean_tweet`, is applied to the raw tweet text. This function performs several cleaning operations:
      * Removes URLs
      * Removes user mentions (e.g., `@username`)
      * Removes the hashtag symbol (`#`)
      * Removes special characters and punctuation
      * Converts the text to lowercase
3.  **Train/Validation Split**: The training data is split into an 80% training set and a 20% validation set. This allows for an unbiased evaluation of the model's performance before generating final predictions.
4.  **Model Pipeline**: A `Pipeline` object from Scikit-learn is constructed to streamline the machine learning process. It consists of two main stages:
      * **TF-IDF Vectorization**: `TfidfVectorizer` converts the cleaned text into a matrix of TF-IDF features. It is configured to use the top 7,000 features and considers both unigrams and bigrams (`ngram_range=(1,2)`).
      * **Classification**: `LinearSVC` (Linear Support Vector Classifier) is used as the classification model. The `class_weight="balanced"` parameter is used to handle potential imbalances in the dataset's labels.
5.  **Model Training & Evaluation**: The pipeline is trained on the 80% training split and then evaluated on the 20% validation split.
6.  **Final Prediction & Submission**: After validation, the pipeline is retrained on the **entire** training dataset. The final model is then used to predict labels for the cleaned test data. The results are saved to a `submission_svm.csv` file in the required format.

## 📊 Model Performance

The model's performance was evaluated on the validation set, achieving an overall **accuracy of 88%**. The detailed classification report is as follows:

```
Validation Report:

              precision    recall  f1-score   support

           0       0.94      0.90      0.92      1152
           1       0.75      0.85      0.80       432

    accuracy                           0.88      1584
   macro avg       0.85      0.88      0.86      1584
weighted avg       0.89      0.88      0.89      1584
```

## 🚀 How to Run

1.  **Prerequisites**: Ensure you have Python and the following libraries installed:

      * `pandas`
      * `scikit-learn`
      * `notebook` (for running the Jupyter Notebook)
      * `google-colab` (if you are running it in the Google Colab environment)

    You can install them using pip:

    ```bash
    pip install pandas scikit-learn notebook google-colab
    ```

2.  **Dataset**: Place the `train.csv` and `test.csv` files in the location specified in the notebook. The current notebook is configured for Google Colab and expects the files in `/content/drive/MyDrive/Colab Notebooks/`. You may need to update the file paths in cell \#4:

    ```python
    train_df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/train.csv")
    test_df = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/test.csv")
    ```

3.  **Execution**: Open `nlp_internship.ipynb` in Jupyter Notebook or Google Colab and run the cells sequentially from top to bottom.

## 📤 Output

The script will generate a file named `submission_svm.csv`, which contains the `id` from the test set and the `label` predicted by the SVM model.