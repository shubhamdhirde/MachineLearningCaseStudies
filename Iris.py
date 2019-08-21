from sklearn import tree
from scipy.spatial import distance
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_spilt

def euc(a,b):
    return distance.euclidean(a,b)

class MLKNN():
    def fit(self,TrainingData,TrainingTarget):
        self.TrainingData = TrainingData
        self.TrainingTarget = TrainingTarget
    
    def predict(self,TestData):
        predictions = []

        for row in TestData:
            lebel = self.closet(row)
            predictions.append(lebel)
        return predictions

    def closet(self,row):
        bestdistance = euc(row,self.TrainingData[0])
        bestindex = 0

        for i in range(1,len(self.TraingData)):
            dist = euc(row,self.TrainingData[i])
            if dist < bestdistance:
                bestdistance = dist
                bestindex = i
        return self.TrainingTarget[bestindex]

def KNeighbourX():
    border = "-"*50

    iris = load_iris()

    data = iris.data
    target = iris.target

    print(border)
    print("Actual dataset")
    print(border)

    for i in range(len(iris.target)):
        print("ID: %d, Label %s, Feature : %s" %(i,iris.data[i],iris.target[i]))
    print("Size of actual dataset %d"%(i+1))

    data_train,data_test,target_train,target_test = train_test_spilt(data,target,test_size = 0.5)

    print(border)
    print("Training dataset")
    print(border)
    for i in range(len(data_train)):
        print("ID: %d, Label %s, Feature : %s" %(i,data.train[i],target_train[i]))
    print("Size of training dataset %d"%(i+1))


    print(border)
    print("Test dataset")
    print(border)
    for i in range(len(data_test)):
        print("ID: %d, Label %s, Feature : %s" %(i,data_test[i],target_test[i]))
    print("Size of test dataset %d"%(i+1))
    print(border)

    classifier = MLKNN()

    classifier.fit(data_train,target_train)

    predictions = classifier.predict(data_test)

    Accuracy = accuracy_score(target_test,predictions)

    return Accuracy

def main():

    Accuracy = KNeighbourX()
    print("Accuracy is ",Accuracy*100,"%")

if __name__ == "__main__":
    main()

