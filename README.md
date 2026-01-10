# ML-Basic
Basic Machine Learning code from Visual Code Studio
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load gene count data from TSV
data = pd.read_csv("gene_counts_.tsv", sep="\t")

print("Initial columns:", data.columns.tolist())
print("Initial index name:", data.index.name)

# Drop 'gene_id' and 'gene_name' columns if present, as they are non-numeric identifiers
for col in ['gene_id', 'gene_name']:
    if col in data.columns:
        data = data.drop(columns=[col])
        print(f"Dropped column: {col}")

# Transpose so rows are samples and columns are genes
data = data.T
print("Shape after transpose (samples x genes):", data.shape)

# Load sample metadata with target labels
labels = pd.read_excel('Coldata.xlsx')
print("Labels shape:", labels.shape)

# Make sure the samples in data and labels match in order
# If your labels dataframe has sample IDs, you can align them here.
# For example:
# data.index.name = 'SampleID'
# data = data.loc[labels['SampleID']]

# Assign target variable (make sure the column name matches your file)
y = labels['Condition']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

# Train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Evaluate and print accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy:.2f}")
