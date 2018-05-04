def getGroundTruth(test_labels,columns):
    groundtruth = []
    for i in range(9):
        if (test_labels[i] == 1):
            groundtruth.append(columns[i + 1])
    return groundtruth

output = []
actual = []

def evaluate(pred,Y_test,columns):
    for i in range(len(Y_test)):
        print(pred[i])
        print(Y_test[i])
        actual.append(getGroundTruth(Y_test[i],columns))
        classes = []
        for j in range(9):
            if pred[i][j] >= 0.5:
                classes.append(columns[j + 1])
        output.append(classes)

    evaluateModel(output,actual)

def calculateMetrics(output,groundtruth,label):
    true_positives = 0.0
    true_negatives = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    for x in groundtruth :
        for i  in output:
            for j in i:
                for y in x:
                    if label in y and label in j:
                        true_positives += 1
                    if label in y and label not in j:
                        false_negatives += 1
                    if label in j and label not in y:
                        false_positives += 1
                    if label not in j and label not in y:
                        true_negatives += 1

    return true_positives,true_negatives,false_positives,false_negatives


def displayMetrics(tp, tn, fp, fn):
    precision = 0.0
    recall = 0.0
    f1 =0.0
    if tp + fp != 0.0:
        precision = tp / (tp + fp)
    if tp + fn != 0.0:
        recall = tp /(tp + fn)
    if (precision+recall) != 0.0:
        f1 = (2*precision*recall)/(precision+recall)

    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 score: ",f1)


def evaluateModel(output,groundtruth):
    # for label seek
    tp,tn,fp,fn = calculateMetrics(output,groundtruth,"Seek")
    print("Metrics for Seek Frame")
    displayMetrics(tp,tn,fp,fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, "medical_condition")
    print("Metrics for medical_condition Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, "medical_test")
    print("Metrics for medical_test Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, "medication")
    print("Metrics for medication Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, "insurance")
    print("Metrics for insurance Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, "diet")
    print("Metrics for diet Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, "exercise")
    print("Metrics for exercise Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, "ask_for_advice")
    print("Metrics for ask_for_advice Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, "other")
    print("Metrics for other Frame")
    displayMetrics(tp, tn, fp, fn)



