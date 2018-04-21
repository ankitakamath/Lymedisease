import json
import os
from collections import Counter
import matplotlib.pyplot as plt

Seek =[ "seeking" ,"Scrutiny"]
Medical_conditions = [ "cause","causation","effect","cause_harm","victim"," Biological_urge","Sign","Catastrophe"
                      "Toxic_substance", "damaging"]
medical_test = ["operational_testing", "medical_professionals", "medical_specialities","medical_instruments"]
Ask_for_advice = [ "opinion","seeking_to_achieve"]
medication  = ["intoxicants","preventing","Medical_specialties","medicines", "medical_intervention","cure"]
insurance = []
exercise = ["observable_body_parts"]
diet =  ["ingredients","food"]
frameList = dict()

"""
Reads the files from the folder
"""
def getfiles(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file


"""
Returns frame details of the relevant frames from the semafor output
"""
def getFrameDetails():
    path = "/home/ankita/RIT_sem4/Project/Frame_Extraction_Tool/output/"
    for file in getfiles(path):
        with open(path + file) as fname:
            for line in fname:
                json_data = json.loads(line)
                frame_set = json_data['frames']
                for frame in frame_set:
                    frame_name = frame["target"]["name"].lower()
                    checkFrame(frame_name)
                    frame_annotations = frame["annotationSets"]
                    for annotation in frame_annotations:
                        frame_elements = annotation["frameElements"]
                        for frame_element in frame_elements:
                            frame_elements_name = frame_element["name"]
                            checkFrame(frame_elements_name)

    c = Counter(frameList)
    frames = []
    count = []
    for k, v in c.items():
        frames.append(k)
        count.append(v)
    drawHistogram(frames,count)


def drawHistogram(frames, countlist):
    x = list(range(1,len(frames)+1))
    plt.bar(x, countlist, align='center')
    plt.xticks(x, frames,fontsize = 14)
    plt.xlabel("Frames",fontsize = 16)
    ylab = "Number of frames"
    plt.ylabel(ylab,fontsize = 16)
    plt.show()

def checkFrame(frame_name):
    if frame_name in Seek:
        if "Seek" in frameList.keys():
            temp = frameList.get("Seek")
            frameList["Seek"] = temp + 1
        else:
            frameList["Seek"] = 1
    elif frame_name in Medical_conditions:
        if "Medical_conditions" in frameList.keys():
            temp = frameList.get("Medical_conditions")
            frameList["Medical_conditions"] = temp + 1
        else:
            frameList["Medical_conditions"] = 1
    elif frame_name in medical_test:
        if "medical_test" in frameList.keys():
            temp = frameList.get("medical_test")
            frameList["medical_test"] = temp + 1
        else:
            frameList["medical_test"] = 1
    elif frame_name in Ask_for_advice:
        if "Ask_for_advice" in frameList.keys():
            temp = frameList.get("Ask_for_advice")
            frameList["Ask_for_advice"] = temp + 1
        else:
            frameList["Ask_for_advice"] = 1
    elif frame_name in exercise:
        if "exercise" in frameList.keys():
            temp = frameList.get("exercise")
            frameList["exercise"] = temp + 1
        else:
            frameList["exercise"] = 1
    elif frame_name in diet:
        if "diet" in frameList.keys():
            temp = frameList.get("diet")
            frameList["diet"] = temp + 1
        else:
            frameList["diet"] = 1
    elif frame_name in medication:
        if "medication" in frameList.keys():
            temp = frameList.get("medication")
            frameList["medication"] = temp + 1
        else:
            frameList["medication"] = 1

def main():
    getFrameDetails()


if __name__ == '__main__':
    main()
