import json
from collections import Counter
import matplotlib.pyplot as plt

def getUserCount():
    userpost = []
    usercomment = []
    with open('medicalQuestionData.json', 'r') as f1, open('seekingDoctorData.json', 'r') as f2, \
            open('generalSupportData.json', 'r') as f3:
        for line in f1:
            json_data = json.loads(line)
            user = json_data["userID"]
            postorComment = json_data["postOrComment"]
            if postorComment == "Post":
                userpost.append(user)
            elif postorComment == "Comment":
                usercomment.append(user)

        for line in f2:
            json_data = json.loads(line)
            user = json_data["userID"]
            if postorComment == "Post":
                userpost.append(user)
            elif postorComment == "Comment":
                usercomment.append(user)

        for line in f3:
            json_data = json.loads(line)
            user = json_data["userID"]
            if postorComment == "Post":
                userpost.append(user)
            elif postorComment == "Comment":
                usercomment.append(user)

        c1 = Counter(usercomment)
        c2 = Counter(userpost)
        # bins = list(range(100,20000))
        bins = [300, 400, 500, 1000, 2000, 3000,4000,6000, 8000, 10000,12000, 15000, 20000]
        plt.hist(list(c1.values()),bins ,alpha=0.5, histtype='bar', ec='black')
        plt.xlabel("Number of comments")
        plt.ylabel("Number of users")
        plt.show()

        bins = [100,200,300,400,500,600]
        plt.hist(list(c2.values()), bins, alpha=0.5, histtype='bar', ec='black')
        plt.xlabel("Number of posts")
        plt.ylabel("Number of users")
        plt.show()


def getText(users):
    adminuser = ["Jenifer", "Lymetoo", "Robin123", "sixgoofykids", "faithful777"]
    #adminuser = ["bettyg","Lymetoo","Siciliano","hopingandpraying","Keebler","Tincup","TF","Abxnomore","mbroderick"]
    textList = []
    with open('medicalQuestionData.json', 'r') as f1,open('seekingDoctorData.json', 'r') as f2, \
            open('generalSupportData.json','r') as f3, open('labelling_posts','w') as output:
        for line in f1:
            json_data = json.loads(line)
            user = json_data["userID"]
            postorComment = json_data["postOrComment"]
            if postorComment == "Post":
                output.write("Post: \n")
            elif postorComment == "Comment":
                output.write("Comment: \n")
            if user in users and user not in adminuser:
                if 'text' in json_data:
                    post = json_data['text'].lower()
                    if not post.strip(): continue
                    textList.append(post)
                    output.write(post + "\n")
                    output.write("\n")

        for line in f2:
            json_data = json.loads(line)
            user = json_data["userID"]
            postorComment = json_data["postOrComment"]
            if postorComment == "Post":
                output.write("Post: \n")
            elif postorComment == "Comment":
                output.write("Comment: \n")
            if user in users and user not in adminuser:
                if 'text' in json_data:
                    post = json_data['text'].lower()
                    if not post.strip(): continue
                    textList.append(post)
                    output.write(post + "\n")
                    output.write("\n")


        for line in f3:
            json_data = json.loads(line)
            user = json_data["userID"]
            postorComment = json_data["postOrComment"]
            if postorComment == "Post":
                output.write("Post: \n")
            elif postorComment == "Comment":
                output.write("Comment: \n")
            if user in users and user not in adminuser:
                if 'text' in json_data:
                    post = json_data['text'].lower()
                    if not post.strip(): continue
                    textList.append(post)
                    output.write(post + "\n")
                    output.write("\n")



    print(len(textList))

def getTopUsers():
    users = []
    with open('medicalQuestionData.json', 'r') as f:
        for line in f:
            json_data = json.loads(line)
            if "userID" in json_data:
                user = json_data['userID']
                users.append(user)

    with open('seekingDoctorData.json', 'r') as f:
        for line in f:
            json_data = json.loads(line)
            if "userID" in json_data:
                user = json_data['userID']
                users.append(user)

    with open('generalSupportData.json','r') as f:
        for line in f:
            json_data = json.loads(line)
            if "userID" in json_data:
                user = json_data['userID']
                users.append(user)

    c = Counter(users)
    users = []
    for k, v in c.most_common(15):
        users.append(k)
    return users

def getPostsByUsers(users):
    userpost=[]
    usercomment = []
    adminuser = ["bettyg", "Lymetoo", "Siciliano", "hopingandpraying", "Keebler", "Tincup", "TF", "Abxnomore",
                 "mbroderick"]
    with open('medicalQuestionData.json', 'r') as f1, open('seekingDoctorData.json', 'r') as f2, \
            open('generalSupportData.json', 'r') as f3:
        for line in f1:
            json_data = json.loads(line)
            user = json_data["userID"]
            postorComment = json_data["postOrComment"]
            if user in users and user not in adminuser:
                if postorComment == "Post":
                    userpost.append(user)
                elif postorComment == "Comment":
                    usercomment.append(user)


        for line in f2:
            json_data = json.loads(line)
            user = json_data["userID"]
            if user in users and user not in adminuser:
                if postorComment == "Post":
                    userpost.append(user)
                elif postorComment == "Comment":
                    usercomment.append(user)


        for line in f3:
            json_data = json.loads(line)
            user = json_data["userID"]
            if user in users and user not in adminuser:
                if postorComment == "Post":
                    userpost.append(user)
                elif postorComment == "Comment":
                    usercomment.append(user)

    c1 = Counter(userpost)
    c2 = Counter(usercomment)
    users1 = []
    postCount1  = []
    for k,v in c1.items():
        users1.append(k)
        postCount1.append(v)

    users2 = []
    postCount2 = []
    for k, v in c2.items():
        users2.append(k)
        postCount2.append(v)

    drawHistogram(users1,postCount1,"Posts")
    drawHistogram(users2,postCount2,"Comments")

def drawHistogram(users, countlist, s):
    x = list(range(1,len(users)+1))
    plt.bar(x, countlist, align='center')
    plt.xticks(x, users)
    plt.xlabel("users")
    ylab = "count of "+ s
    plt.ylabel(ylab)
    plt.show()

def main():
    users = getTopUsers()
    getText(users)
    # getUserCount()


if __name__ == '__main__':
    main()
