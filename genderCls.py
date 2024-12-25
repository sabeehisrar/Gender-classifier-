from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Training data: [height, weight, shoe size]
X = [[175, 70, 42],
     [160, 55, 37],
     [180, 80, 44],
     [165, 60, 38],
     [170, 65, 40],
     [155, 50, 36],
     [185, 85, 45],
     [158, 52, 35],
     [172, 68, 41],
     [162, 58, 39],
     [177, 73, 43]]

# Labels
Y = ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male']

# Test sample: [height, weight, shoe size]
test_sample_male = [[195, 80, 40]]

# Models
models = {
    "Decision Tree": tree.DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
    "Support Vector Machine": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=10),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression()
}

# Train and test each model
for model_name, model in models.items():
    model.fit(X, Y)
    prediction = model.predict(test_sample_male)
    print(f"Prediction via {model_name}: {prediction[0]}")

