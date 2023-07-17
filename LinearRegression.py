dataFile = open("AdvertisingData.csv", "r")
dataFile.readline()
dataLines = dataFile.readlines()
data = []
for line in dataLines:
   splits = line.split(",")
   spending = (float)(splits[1]) + (float)(splits[2]) + (float)(splits[3])
   sales = (float)(splits[4])
   data.append((spending, sales))

dataFile.close()

print(len(data))

def predict(w, b, newFeature):
    return w * newFeature + b
    
# theory
def sumFeatures(data):
    sum = 0.0
    for i in range(len(data)):
        sum += data[i][0]
    return sum

def sumLabels(data):
    sum = 0.0
    for i in range(len(data)):
        sum += data[i][1]
    return sum
 
# function l 
def calculateEmpiricalRisk(data, w, b):
    averageLoss = 0.0
    for i in range(len(data)):
        feature = data[i][0]
        label = data[i][1]
        averageLoss += pow(predict(w, b, feature) - label, 2) / len(data)
    return averageLoss

def calculate_dl_dw(data, w, b):
    dl_dw = 0.0 #(sumLabels(data) - len(data) * b) / sumFeatures(data)
    for i in range(len(data)):
        feature = data[i][0]
        label = data[i][1]
        dl_dw += 2 * feature * (predict(w, b, feature) - label) / len(data)
        
    return dl_dw

def calculate_dl_db(data, w, b):
    dl_db = 0.0 #(sumLabels(data) - w * sumFeatures(data)) / len(data)
    for i in range(len(data)):
        feature = data[i][0]
        label = data[i][1]
        dl_db += 2 * (predict(w, b, feature) - label) / len(data)
        
    return dl_db

def learnModel(data, w, b, learningRate):
    dw = learningRate * calculate_dl_dw(data, w, b)
    db = learningRate * calculate_dl_db(data, w, b)
    return w - dw, b - db

def reportEpoch(data, w, b, currentEpoch):
    print("Epoch " + str(currentEpoch) + " (loss: " + str(calculateEmpiricalRisk(data, w, b)) + ", w: " + str(w) + ", b: " + str(b) + ")")
    
def train(data, w, b, learningRate, numEpochs):
    for i in range(numEpochs):
        if (i % 500 == 0):
            reportEpoch(data, w, b, i)
        w, b = learnModel(data, w, b, learningRate)
    
    reportEpoch(data, w, b, numEpochs)
    return w, b

w = sumLabels(data) / sumFeatures(data)
b = 0
w, b = train(data, w, b, 0.00002, 10000)
print(predict(w, b, 250))