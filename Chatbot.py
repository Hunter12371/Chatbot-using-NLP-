import json
import string
import random
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer
import tensorflow as tensorF
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download("punkt")
nltk.download("wordnet")

# Creating JSON data that lists all possible outcomes of the user interaction with our bot
ourData = {"intents": [
    {"tag": "age",
     "patterns": ["how old are you?"],
     "responses": ["I was just born recently and learning everything"]
     },
    {"tag": "greeting",
     "patterns": ["Hi", "Hello", "Hey"],
     "responses": ["Hi there", "Hello", "Hi :)"],
     },
    {"tag": "goodbye",
     "patterns": ["bye", "later"],
     "responses": ["Bye", "take care"]
     },
    {"tag": "name",
     "patterns": ["what's your name?", "who are you?"],
     "responses": ["I have no name yet", "You can give me one, and I will appreciate it"]
     }
]}

# Now we will process the data before creating our training data

lm = WordNetLemmatizer()  # It is used for getting our words

ourClasses = []
newWords = []
documentX = []
documentY = []

# Fixed: Changed data to ourData
for intent in ourData["intents"]:
    for pattern in intent["patterns"]:
        ournewTkns = nltk.word_tokenize(pattern)
        newWords.extend(ournewTkns)
        documentX.append(pattern)
        documentY.append(intent["tag"])

    if intent["tag"] not in ourClasses:
        ourClasses.append(intent["tag"])

newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation]
newWords = sorted(set(newWords))
ourClasses = sorted(set(ourClasses))

# Now we will design our neural network for the Chatbot
# It is created using the Bag of Words (BoW) encoding system

trainingData = []
outEmpty = [0] * len(ourClasses)

# BoW Model

for idx, doc in enumerate(documentX):
    bagOfwords = []
    text = lm.lemmatize(doc.lower())
    for word in newWords:
        bagOfwords.append(1) if word in text else bagOfwords.append(0)

    outputRow = list(outEmpty)
    outputRow[ourClasses.index(documentY[idx])] = 1
    trainingData.append([bagOfwords, outputRow])

random.shuffle(trainingData)
trainingData = num.array(trainingData, dtype=object)

x = num.array(list(trainingData[:, 0]))  # This is the First Training Phase
y = num.array(list(trainingData[:, 1]))  # This is the Second Training Phase

iShape = (len(x[0]),)
oShape = len(y[0])

ourNewModel = Sequential()

ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))

ourNewModel.add(Dropout(0.5))

ourNewModel.add(Dense(64, activation="relu"))
ourNewModel.add(Dropout(0.3))
ourNewModel.add(Dense(oShape, activation="softmax"))

md = tensorF.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)

ourNewModel.compile(loss='categorical_crossentropy',
                    optimizer=md,
                    metrics=["accuracy"])

print(ourNewModel.summary())

ourNewModel.fit(x, y, epochs=200, verbose=1)  # By epochs, It means the number of times you can repeat a training set.


# Now we will build useful features for the Chatbot

def ourText(text):
    newtkns = nltk.word_tokenize(text)
    newtkns = [lm.lemmatize(word) for word in newtkns]
    return newtkns


def wordBag(text, vocab):
    newtkns = ourText(text)
    bagOwords = [0] * len(vocab)
    for w in newtkns:
        for idx, word in enumerate(vocab):
            if word == w:
                bagOwords[idx] = 1
    return num.array(bagOwords)


def Pclass(text, vocab, labels):
    bagOwords = wordBag(text, vocab)
    # Fixed: Added reshape to handle input shape
    ourResult = ourNewModel.predict(num.array([bagOwords]), verbose=0)[0]
    newThresh = 0.2
    yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

    yp.sort(key=lambda x: x[1], reverse=True)
    newList = []
    for r in yp:
        newList.append(labels[r[0]])
    return newList


def getRes(firstlist, fJson):
    # Added error handling in case no intents match
    if not firstlist:
        return "I'm not sure how to respond to that."

    tag = firstlist[0]
    listOfIntents = fJson["intents"]
    for i in listOfIntents:
        if i["tag"] == tag:
            ourResult = random.choice(i["responses"])
            break
    return ourResult


# And finally inputting these values

print("Your chatbot is ready! Type something to start a conversation (or type 'quit' to exit):")
while True:
    newMessage = input("You: ")
    if newMessage.lower() == 'quit':
        print("Bot: Goodbye!")
        break

    intents = Pclass(newMessage, newWords, ourClasses)
    ourResult = getRes(intents, ourData)
    print("Bot:", ourResult)
