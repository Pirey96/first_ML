import numpy as np
import pandas as pd
from sklearn import metrics
import string
import matplotlib.pyplot as plot
import math
data_frame = pd.read_csv("hns_2018_2019.csv")
#lists to make it easy to use pandas
Data = []          #complete dataset
Post_type = []       #stores the posttype
Title = []          #stores title
Year = []             #stores the year written
txt_list = []
word=[]
t_set = []
new_set = []
voc = 0
txt_list_new = []
#This function gets all the data for 2018
def get_data():
   Title = data_frame.Title
   Post_type = data_frame['Post Type']
   Year = data_frame.year
   for i in range(len(Title)):
       if int(Year[i]) == 2018:
           Data.append( [Title[i],Post_type[i]])
       elif int(Year[i])==2019:
           t_set.append([Title[i],Post_type[i]])

story = 0
ask = 0
show = 0
poll = 0

def classify_words():
    # count to keep track of nb of words
    count = 2
    # for counting the number of words in each class
    global story
    global ask
    global show
    global poll
    global word

    for i in Data:

        #configure the stirngs to isolate all the words
        #to accomplish this the replace method is used
        #to replace all excess characters such as commas
        #question marks and other chars that would differntiate same words

        i[0] = i[0].lower()
        i[0] = i[0].replace("?", " QUESTIONMARK ")
        i[0] = i[0].replace("?", " EXCLAMATIONMARK ")
        "".join(char for char in i[0] if char not in string.punctuation)
        "".join(char for char in i[0] if char not in string.punctuation)


        #where the actual splitting happens

        temp = i[0].split()


        #this function appends new words to the vocabulary and adds words that reappear in the approcpriate cell
        #if its a story, the story increments(etc)
        for j in range(len(temp)):
            if len(txt_list) == 0:
                if i[1] == 'story':
                    story = story + 1
                    txt_list.append([1, temp[j], story, 0, ask, 0, show, 0, poll, 0])
                    word.append(temp[j])
                    continue
                elif i[1] == 'ask_hn':
                    ask = ask +1
                    txt_list.append([1, temp[j], story, 0, ask, 0, show, 0, poll, 0])
                    word.append(temp[j])
                    continue
                elif i[1] == 'show_hn':
                    show = show +1
                    txt_list.append([1, temp[j], story, 0, ask, 0, show, 0, poll, 0])
                    word.append(temp[j])
                    continue
                elif i[1] == 'poll':
                    poll = poll +1
                    txt_list.append([1, temp[j], story, 0, ask, 0, show, 0, poll, 0])
                    word.append(temp[j])
                    continue
            if temp[j] not in word:
                 if i[1] == 'story':
                     story = story + 1
                     txt_list.append([count, temp[j], 1, 0, 0, 0, 0, 0, 0, 0])
                     word.append(temp[j])
                     count = count + 1
                     continue
                 elif i[1] == 'ask_hn':
                     ask = ask + 1
                     txt_list.append([count, temp[j], 0, 0, 1, 0, 0, 0, 0, 0])
                     count = count + 1
                     word.append(temp[j])
                     continue
                 elif i[1] == 'show_hn':
                     show = show + 1
                     txt_list.append([count, temp[j], 0, 0, 0, 0, 1, 0, 0, 0])
                     count = count + 1
                     word.append(temp[j])
                     continue
                 elif i[1] == 'poll':
                     poll = poll + 1
                     txt_list.append([count, temp[j], 0, 0, 0, 0, 0, 0, 1, 0])
                     count = count +1
                     word.append(temp[j])
                     continue
            else:

                for data in txt_list:
                    if data[1] == temp[j]:
                        if i[1]=='story':
                            story = story + 1
                            data[2] = data[2]+1
                        elif i[1] == 'ask_hn':
                            data[4]=data[4]+1
                            ask = ask + 1
                        elif i[1]=='show_hn':
                            data[6]=data[6]+1
                            show = show + 1
                        elif i[1] == 'poll':
                            poll = poll + 1
                            data[8]=data[8]+1
    voc = story+poll+ask+show
    #print ("Story: "+str(story)+'\n'+"ask_hn: "+str(ask)+"\n"+"show_hn: "+str(show)+"\n"+"poll: "+str(poll))
    smoothed_bayes()



def smoothed_bayes():
    #smoothed conditonal probabilies
    #conditional probabilty variables
    for rows in txt_list:
        voc = story+ask+show+poll
        #smoothed values for each variable
        smoothed_story = rows[2]+0.5
        smoothed_ask = rows[4]+0.5
        smoothed_show = rows[6]+0.5
        smoothed_poll = rows[8]+0.5
        rows[3] = smoothed_story/(story+len(word)*0.5)
        rows[5] = smoothed_ask/(ask+len(word)*0.5)
        rows[7] = smoothed_show/(show+len(word)*0.5)
        rows[9] = smoothed_poll/(poll+len(word)*0.5)



def write_to_vocabulary():
    #write conditional probablity into a txt file
    with open("vocabulary.txt", "w", encoding='utf-8') as text_file:
        for i in txt_list:
            text_file.write(str(i)+"\n")
#an array for the test words to be computed
right = 0

def test_set():
    count = 0
    cst= 0
    cak = 0
    csh = 0
    cpo = 0
    cst1 = 0
    cak1 = 0
    csh1= 0
    cpo1 = 0
    all = 0
    with open("baseline-result.txt", 'w', encoding='utf-8') as text_file:
        for t in t_set:
            t[0] = t[0].lower()
            t[0] = t[0].replace("?"," QUESTION")
            t[0]
            "".join(char for char in t[0] if char not in string.punctuation)

            temp = t[0].split()
            prob_story = []
            prob_ask = []
            prob_show = []
            prob_poll = []
            prob_story.append(np.math.log10((story) / (story + ask + poll + show)))
            prob_ask.append(np.math.log10((ask) / (story + ask + poll + show)))
            prob_show.append(np.math.log10((show) / (story + ask + poll + show)))
            prob_poll.append(np.math.log10((poll + 0.5 )/ (story + ask + poll + show)))
            all = all+1
            for test_words in temp:


                if test_words in word:
                    index = txt_list[word.index(test_words)]
                    prob_story.append(np.math.log10(index[3]))
                    prob_ask.append(np.math.log10(index[5]))
                    prob_show.append(np.math.log10(index[7]))
                    prob_poll.append(np.math.log10(index[9]))

                else:
                    continue

            #decides the likely category the given title beliongs to
            #using a naive bayes classifier and logarithms to avoid
            #arithmetic underflows
            st = sum(prob_story)
            sh = sum(prob_show)
            ak = sum(prob_ask)
            po = sum(prob_poll)
            highest = max(st, sh, ak, po)

            global right
            if highest == st:
                prediction = 'story'
            elif highest == ak:
                prediction='ask_hn'
            elif highest == sh:
                prediction = 'show_hn'
            else:
                prediction = 'poll'
            if prediction == t[1]:
                correctness = "right"
                right = right+1
            else:
                correctness = 'wrong'
            op = []
            count = count+1
            op.append([count, t[0], prediction, st, ak, sh, po, t[1], correctness])
            #count the number of predicted stories, polls, asks and shows
            #and counts number actual news stories to compare
            text_file.write(str(op)+"\n")
            if prediction == "story":
                cst=cst+1
            if(t[1] == 'story'):
                cst1 = cst1+1
            if prediction == "ask_hn":
                cak=cak+1
            if t[1] == 'ask_hn':
                cak1 = cak1+1
            if prediction == "show_hn":
                csh=csh+1
            if t[1] == 'show_hn':
                csh1 = csh1+1
            if prediction == "poll":
                cpo=cpo+1
            if(t[1] == 'poll'):
                cpo1 = cpo1+1
    print("Prediction: " + "Story: " + str(cst) + " Show_hn: " + str(csh) + " Ask_hn: " + str(cak)+" Poll: " + str(cpo))
    print("Actual:     " + "Story: " + str(cst1) + " Show_hn: " + str(csh1) + " Ask_hn: " + str(cak1)+" Poll: " + str(cpo1))
    print("Correct predictions "+"(out of "+str(all)+") "+str(right))
    for i in txt_list:
        txt_list_new.append(i)


get_data()
classify_words()
temp_story = story
temp_ask=ask
temp_show=show
temp_poll=poll
write_to_vocabulary()

test_set()



#####SECTIOn 1.3
#experiment 1.3.1
def test_set_exp():
    count = 0
    cst= 0
    cak = 0
    csh = 0
    cpo = 0
    cst1 = 0
    cak1 = 0
    csh1= 0
    cpo1 = 0
    with open("stopword-result.txt", 'w', encoding='utf-8') as text_file:
        for t in t_set:
            t[0] = t[0].lower()
            t[0] = t[0].replace("?", " QUESTIONMARK ")
            t[0] = t[0].replace("!","EXCLAMATIONMARK ")
            "".join(char for char in t[0] if char not in string.punctuation)

            temp = t[0].split()
            prob_story = []
            prob_ask = []
            prob_show = []
            prob_poll = []
            prob_story.append(np.math.log10((story) / (story + ask + poll + show)))
            prob_ask.append(np.math.log10((ask) / (story + ask + poll + show)))
            prob_show.append(np.math.log10((show) / (story + ask + poll + show)))
            prob_poll.append(np.math.log10((poll + 0.5 )/ (story + ask + poll + show)))
            for test_words in temp:

                if test_words in word:
                    index = txt_list[word.index(test_words)]
                    prob_story.append(np.math.log10(index[3]))
                    prob_ask.append(np.math.log10(index[5]))
                    prob_show.append(np.math.log10(index[7]))
                    prob_poll.append(np.math.log10(index[9]))
                else:
                    continue

            #decides the likely category the given title beliongs to
            #using a naive bayes classifier and logarithms to avoid
            #arithmetic underflows
            st = sum(prob_story)
            sh = sum(prob_show)
            ak = sum(prob_ask)
            po = sum(prob_poll)
            highest = max(st, sh, ak, po)

            global right
            if highest == st:
                prediction = 'story'
            elif highest == ak:
                prediction='ask_hn'
            elif highest == sh:
                prediction = 'show_hn'
            else:
                prediction = 'poll'
            if prediction == t[1]:
                correctness = "right"
                right = right+1
            else:
                correctness = 'wrong'
            op = []
            count = count+1
            op.append([count, t[0], prediction, st, ak, sh, po, t[1], correctness])
            #count the number of predicted stories, polls, asks and shows
            #and counts number actual news stories to compare
            text_file.write(str(op)+"\n")
            if prediction == "story":
                cst=cst+1
            if(t[1] == 'story'):
                cst1 = cst1+1
            if prediction == "ask_hn":
                cak=cak+1
            if t[1] == 'ask_hn':
                cak1 = cak1+1
            if prediction == "show_hn":
                csh=csh+1
            if t[1] == 'show_hn':
                csh1 = csh1+1
            if prediction == "poll":
                cpo=cpo+1
            if(t[1] == 'poll'):
                cpo1 = cpo1+1
    print("Prediction: " + "Story: " + str(cst) + " Show_hn: " + str(csh) + " Ask_hn: " + str(cak)+" Poll: " + str(cpo))
    print("Actual:     " + "Story: " + str(cst1) + " Show_hn: " + str(csh1) + " Ask_hn: " + str(cak1)+" Poll: " + str(cpo1))
    print("Correct predictions (out 5000): "+str(right))

def experiment_1():
    #obtain all stop words and put them in a list
    stopwords = open('stopwords.txt','r')
    global txt_list
    global story
    global ask
    global show
    global poll
    count = 0
    sw = stopwords.readlines()
    #removing stopwords from the main array
    for i in sw:
        i = i.strip()
        if i in word:
            story = story - txt_list[word.index(i)][2]
            ask = ask - txt_list[word.index(i)][4]
            show = show-txt_list[word.index(i)][6]
            poll = poll-txt_list[word.index(i)][8]

            del txt_list[word.index(i)]
            del word [word.index(i)]
    #write to the correct folder
    with open("stopword-model.txt", 'w', encoding='utf-8') as text_file:
        for i in txt_list:
            count = count+1
            i[0] = count
            text_file.write(str(i)+"\n")
    global right
    right = 0
    #reimplement naive bayes using the newly trained algorithm
    smoothed_bayes()
    test_set_exp()


print("Stopword experiment: ")
experiment_1()

#EXPERIMENT 1.3.2
def the_ole_switcheroo():
    global txt_list
    txt_list = []
    for i in txt_list_new:
        txt_list.append(i)
    global word
    word = []
    for i in txt_list:
        word.append(i[1])
    global story
    story = temp_story
    global ask
    ask = temp_ask
    global show
    show = temp_show
    global poll
    poll = temp_poll

def test_set_exp2():
    count = 0
    cst= 0
    cak = 0
    csh = 0
    cpo = 0
    cst1 = 0
    cak1 = 0
    csh1= 0
    cpo1 = 0
    with open("stopword-result.txt", 'w', encoding='utf-8') as text_file:
        for t in t_set:
            t[0] = t[0].lower()
            t[0] = t[0].replace("?", " QUESTIONMARK ")
            t[0] = t[0].replace("!", "EXCLAMATIONMARK ")
            "".join(char for char in t[0] if char not in string.punctuation)

            temp = t[0].split()
            prob_story = []
            prob_ask = []
            prob_show = []
            prob_poll = []
            prob_story.append(np.math.log10((story) / (story + ask + poll + show)))
            prob_ask.append(np.math.log10((ask) / (story + ask + poll + show)))
            prob_show.append(np.math.log10((show) / (story + ask + poll + show)))
            prob_poll.append(np.math.log10((poll + 0.5 )/ (story + ask + poll + show)))
            for test_words in temp:

                if test_words in word:
                    index = txt_list[word.index(test_words)]
                    prob_story.append(np.math.log10(index[3]))
                    prob_ask.append(np.math.log10(index[5]))
                    prob_show.append(np.math.log10(index[7]))
                    prob_poll.append(np.math.log10(index[9]))
                else:
                    continue

            #decides the likely category the given title beliongs to
            #using a naive bayes classifier and logarithms to avoid
            #arithmetic underflows
            st = sum(prob_story)
            sh = sum(prob_show)
            ak = sum(prob_ask)
            po = sum(prob_poll)
            highest = max(st, sh, ak, po)

            global right
            if highest == st:
                prediction = 'story'
            elif highest == ak:
                prediction='ask_hn'
            elif highest == sh:
                prediction = 'show_hn'
            else:
                prediction = 'poll'
            if prediction == t[1]:
                correctness = "right"
                right = right+1
            else:
                correctness = 'wrong'
            op = []
            count = count+1
            op.append([count, t[0], prediction, st, ak, sh, po, t[1], correctness])
            #count the number of predicted stories, polls, asks and shows
            #and counts number actual news stories to compare
            text_file.write(str(op)+"\n")
            if prediction == "story":
                cst=cst+1
            if(t[1] == 'story'):
                cst1 = cst1+1
            if prediction == "ask_hn":
                cak=cak+1
            if t[1] == 'ask_hn':
                cak1 = cak1+1
            if prediction == "show_hn":
                csh=csh+1
            if t[1] == 'show_hn':
                csh1 = csh1+1
            if prediction == "poll":
                cpo=cpo+1
            if(t[1] == 'poll'):
                cpo1 = cpo1+1
    print("Prediction: " + "Story: " + str(cst) + " Show_hn: " + str(csh) + " Ask_hn: " + str(cak)+" Poll: " + str(cpo))
    print("Actual:     " + "Story: " + str(cst1) + " Show_hn: " + str(csh1) + " Ask_hn: " + str(cak1)+" Poll: " + str(cpo1))
    print("Correct predictions (out 5000): "+str(right))
def experiment_2():
    the_ole_switcheroo()
    global story
    global ask
    global show
    global poll
    for i in txt_list:
        if len(i[1])>=9 or len(i[1])<=2:

            story = story-i[2]
            ask = ask-i[4]
            show = show-i[6]
            poll=poll-i[8]
            txt_list.remove(i)

    for i in word:
        if len(i)>=9 or len(i)<=2:
            word.remove(i)
    count = 0
    print("Lengthword experiment: ")
    with open ("wordlength-model.txt","w",encoding="utf-8") as text_file:
        for i in txt_list:
            count=count+1
            i[0]=count
            text_file.write(str(i)+"\n")
    global right
    right = 0
    smoothed_bayes()
    test_set_exp2()

experiment_2()


#PERFORMANCE EXPERIMENT
def plotting():

    bottom = [1,5,10,15,20]
    global right
    xb = []
    yb = []
    for i in range(5):
        right = 0

        for j in txt_list:
            if bottom[i]>=(j[2]+j[4]+j[6]+j[8]):
                txt_list.remove(j)
                word.remove(j[1])
        test_set()
        xb.append(right)
        yb.append(len(txt_list))
    ax = plot.subplot(1,2,1)
    ax.set_title("Removed words <1,5,10,15,20")
    plot.plot(xb,yb)

    top = [5,10,15,20,25]
    the_ole_switcheroo()
    word_num = []
    for i in txt_list:
        word_num.append(i[2]+i[4]+i[6]+i[8])
        list.sort(word_num)
    xt = []
    yt = []
    for i in range(5):
        right = 0
        percent = (100-math.ceil((top[i]/100))*len(word_num))
        for j in txt_list:
            if word_num[percent]<=(j[2]+j[4]+j[6]+j[8]):
                txt_list.remove(j)
                word.remove(j[1])
        test_set()
        xt.append(right)
        yt.append(len(word))
    print(len(word))
    ax1 = plot.subplot(1,2,2)
    ax1.set_title("Removed words top (5,10,15,20,25) percentiles")
    plot.plot(xt,yt)

    plot.show()

plotting()



#CODE FOR DEMO
def new_data_classify():
    global t_set
    t_set = []
    global right
    right = 0
    df = pd.read_csv("newdata.csv")
    new_t= df.Title
    new_pt = df['Post Type']
    for i in range(len(new_t)):
        t_set.append([str(new_t[i]), new_pt[i]])

    test_set()
nextphase = input("Press Enter for next phase ")
#function call below activates this function
#caution this file may not exist and thus may cause errors if called
new_data_classify()

