# Fake_News_Detection

# This is the code of fake news detection and we use random forest classifier and randomisedsearcclassifier for tunning 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score

# Load the dataset
data = pd.read_csv(r"E:\PROYGAM_2024\cleaned_dataset.csv") # change path with your dataset path 

# Split the data into training and testing sets
X = data['title']
y = data['label']

# Vectorize the text data
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_X = tfidf_vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_X, y, test_size=0.2, random_state=42)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Create a Random Forest Classifier with specified hyperparameters
rf_classifier = RandomForestClassifier(
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=4,
    n_estimators=80,
    random_state=42
)

# Train the model on the SMOTE-resampled data
rf_classifier.fit(X_train_smote, y_train_smote)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
precision = precision_score(y_test, y_pred, pos_label='FAKE')
recall = recall_score(y_test, y_pred, pos_label='FAKE')
conf_matrix = confusion_matrix(y_test, y_pred)

# Compute accuracy on training set
y_train_pred = rf_classifier.predict(X_train_smote)
train_accuracy = accuracy_score(y_train_smote, y_train_pred)

# Compute accuracy on testing set
test_accuracy = accuracy_score(y_test, y_pred)

# Print the results
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Training Accuracy: {train_accuracy}')
print(f'Testing Accuracy: {test_accuracy}')
