import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

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
        i[0] = i[0].replace(" ' ","")
        i[0] = i[0].replace("(","")
        i[0] = i[0].replace(")","")
        i[0] = i[0].replace(":","")
        i[0] = i[0].replace("?"," ? ")
        i[0] = i[0].replace(",","")
        i[0] = i[0].replace(".","")
        i[0] = i[0].replace("]","")
        i[0] = i[0].replace("[","")
        i[0] = i[0].replace("!"," ! ")
        i[0] = i[0].replace('/'," ")
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
    smoothed_bayes(story,ask,show,poll)



def smoothed_bayes(story,ask,show,poll):
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
    with open("baseline-result.txt", 'w', encoding='utf-8') as text_file:
        for t in t_set:
            t[0] = t[0].lower()
            t[0] = t[0].replace(" ' ","")
            t[0] = t[0].replace("(","")
            t[0] = t[0].replace(")","")
            t[0] = t[0].replace(":","")
            t[0] = t[0].replace("?"," ? ")
            t[0] = t[0].replace(",","")
            t[0] = t[0].replace(".","")
            t[0] = t[0].replace("]","")
            t[0] = t[0].replace("[","")
            t[0] = t[0].replace("!"," ! ")
            t[0] = t[0].replace('/'," ")
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


get_data()
classify_words()
write_to_vocabulary()
test_set()
