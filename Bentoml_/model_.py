# import the IrisClassifier class defined above
from bento_pack import IrisClassifier  ## class load 후 패키징
from sklearn import svm
from sklearn import datasets

# Load training data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Model Training
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# Create a iris classifier service instance
iris_classifier_service = IrisClassifier()

### svm 모델이 irisClassifier와 'model' 로 패키징됨
# Pack the newly trained model artifact
iris_classifier_service.pack('model', clf)


## 경로를 저장해줌
# Save the prediction service to disk for model serving
saved_path = iris_classifier_service.save()
