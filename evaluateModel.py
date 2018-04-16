def calculateMetrics(output,groundtruth,label):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for x in groundtruth :
        for i  in output:
            j = i.split(",")
            y  = x.split(",")

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
    tpr  = 0
    fpr = 0
    precision = 0
    recall = 0
    if tp + fn  is not 0:
        tpr = tp / (tp + fn)
        recall = tp / (tp + fn)
    if fp + tn is not 0:
        fpr = fp / (fp + tn)
    if tp + fp is not 0:
        precision = tp / (tp + fp)

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print("True positive rate: ", tpr)
    print("False Postive rate: ", fpr)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("Accuracy: ", accuracy)


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

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, "progress")
    print("Metrics for progress Frame")
    displayMetrics(tp, tn, fp, fn)

    tp, tn, fp, fn = calculateMetrics(output, groundtruth, "failure")
    print("Metrics for failure Frame")
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



