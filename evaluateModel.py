def calculateMetrics(output,groundtruth,label):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for x in groundtruth :
        for i  in output:
            j = i.split(",")
            y  = x.split(",")
            if label in y and label in output:
                true_positives += 1
            if label in y and label not in output:
                false_negatives += 1
            if label in output and label not in y:
                false_positives += 1
            if label not in output and label not in y:
                true_negatives += 1

    return true_positives,true_negatives,false_positives,false_negatives

# , "medical_test", "medication", "progress", "failure", "insurance",
#                "diet", "exercise", "ask_for_advice", "other"
def evaluateModel(output,groundtruth):
    # for label seek
    tp_seek,tn_seek,fp_seek,fn_seek = calculateMetrics(output,groundtruth,"Seek")

    tpr_seek = tp_seek / (tp_seek + fn_seek)
    fpr_seek = fp_seek / (fp_seek + tn_seek)
    precision_seek = tp_seek / (tp_seek + fp_seek)
    recall_seek = tp_seek / (tp_seek + fn_seek)

    print("Metrics Report for Seek Frame")
    print("True positive rate: ",tpr_seek)
    print("False Postive rate: ",fpr_seek)
    print("Precision: ", precision_seek)
    print("Recall: ",recall_seek)

    # for label "medical_condition"
    tp_med_con, tn_med_con, fp_med_con, fn_med_con = calculateMetrics(output, groundtruth, "medical_condition")

    tpr_med_con = tp_med_con / (tp_med_con + fn_med_con)
    fpr_med_con = fp_med_con / (fp_med_con + tn_med_con)
    precision_med_con = tp_med_con / (tp_med_con + fp_med_con)
    recall_med_con = tp_med_con / (tp_med_con + fn_med_con)

    print("Metrics Report for Seek Frame")
    print("True positive rate: ", tpr_med_con)
    print("False Postive rate: ", fpr_med_con)
    print("Precision: ", precision_med_con)
    print("Recall: ", recall_med_con)

    # # for label seek
    # tp_seek, tn_seek, fp_seek, fn_seek = calculateMetrics(output, groundtruth, "Seek")
    #
    # tpr_seek = tp_seek / (tp_seek + fn_seek)
    # fpr_seek = fp_seek / (fp_seek + tn_seek)
    #
    # # for label seek
    # tp_seek, tn_seek, fp_seek, fn_seek = calculateMetrics(output, groundtruth, "Seek")
    #
    # tpr_seek = tp_seek / (tp_seek + fn_seek)
    # fpr_seek = fp_seek / (fp_seek + tn_seek)
    #
    # # for label seek
    # tp_seek, tn_seek, fp_seek, fn_seek = calculateMetrics(output, groundtruth, "Seek")
    #
    # tpr_seek = tp_seek / (tp_seek + fn_seek)
    # fpr_seek = fp_seek / (fp_seek + tn_seek)
    #

