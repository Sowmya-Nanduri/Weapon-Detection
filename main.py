from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import ttk
from tkinter import filedialog
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout, Flatten
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os
import cv2
from sklearn.naive_bayes import GaussianNB
import joblib
from skimage.transform import resize
from skimage.io import imread
from skimage import io, transform
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
import pickle
from keras.models import model_from_json
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

main = Tk()
main.title("weapon detection")
main.geometry("1300x1200")

global filename
global X, Y
global model
global accuracy
global rf_classifier

shapes =['sword','Sniper','SMG','shotgun','Handgun','Grenadelauncher','Bazooka','AutomaticRifle']


def getID(name):
    if name == 'sword':
        return 0
    elif name == 'Sniper':
        return 1
    elif name == 'SMG':
        return 2
    elif name == 'shotgun':
        return 3
    elif name == 'Handgun':
        return 4
    elif name == 'Grenadelauncher':
        return 5  # Update this line with the correct index
    elif name == 'Bazooka':
        return 6
    elif name == 'AutomaticRifle':
        return 7
    else:
        return -1  # Return an invalid index or handle the unknown category appropriately

    
def uploadDataset():
    global X, Y
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,'dataset loaded\n')
    
def imageProcessing():
    text.delete('1.0', END)
    global X, Y,X_train,X_test,Y_train,Y_test
    X = []
    Y = []
    '''
    for root, dirs, directory in os.walk(filename):
        for j in range(len(directory)):
            name = os.path.basename(root)
            print(name + " " + root + "/" + directory[j])
            
            if 'Thumbs.db' not in directory[j]:
                img_path = os.path.join(root, directory[j])

                try:
                    # Try to read the image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Error reading image: {img_path}")
                        continue  # Skip this image and continue with the next one

                    # Resize the image
                    img = cv2.resize(img, (64, 64))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(64, 64, 3)
                    X.append(im2arr)
                    Y.append(getID(name))
                    
                except Exception as e:
                    print(f"Error processing image: {img_path}. {e}")
                    continue  # Skip this image and continue with the next one

    if not X or not Y:
        text.insert(END, "No valid images found in the dataset.\n")
        return

    X = np.asarray(X)
    Y = np.asarray(Y)
    print(Y)

    X = X.astype('float32')
    X = X / 255
    test = X[3]
    test = cv2.resize(test, (400, 400))
    cv2.imshow("aa", test)
    cv2.waitKey(0)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    np.save('model/X.txt', X)
    np.save('model/Y.txt', Y)
    '''
    text.delete('1.0', END)
    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')
    text.insert(END, "Total number of images found in dataset is  : " + str(len(X)) + "\n")
    text.insert(END, "Total classes found in dataset is : " + str(shapes) + "\n")

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20,random_state=77)
    print(Y_test)
    print(Y_test.shape)
def gnb():
    global x_test,x_train,y_test,y_train,df,p,r,f,acc
    global classifier
    global accuracy
    #text.delete('1.0', END)
    Categories =['sword','Sniper','SMG','shotgun','Handgun','Grenadelauncher','Bazooka','AutomaticRifle']
    flat_data_arr = [] #input array
    target_arr = [] #output array
    datadir = r"data"
    flat_data_file = os.path.join(datadir, 'flat_data.npy')
    target_file = os.path.join(datadir, 'target.npy')
    if os.path.exists(flat_data_file) and os.path.exists(target_file):
        # Load the existing arrays
        flat_data = np.load(flat_data_file)
        target = np.load(target_file)
        #dataframe
        df = pd.DataFrame(flat_data)
        df['Target'] = target #associated the numerical representation of the category (index) with the actual image data
        df
        #input data
        
        x = df.iloc[:,:-1]
        #output data
        y = df.iloc[:,-1]
        # Splitting the data into training and testing sets
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,random_state=0)
        # Load the model from the pkl file
        nb_classifier=GaussianNB()
        nb_classifier.fit(x_train,y_train)
        #nb_classifier = joblib.load('model/Naive_Bayes_model_weights.pkl')
        y_pred = nb_classifier.predict(x_test)
        acc = accuracy_score(y_test, y_pred)*100
        cm = confusion_matrix(y_test, y_pred)
        clr = classification_report(y_test, y_pred, target_names=shapes)
        p = precision_score(y_test, y_pred, average='macro') * 100
        r = recall_score(y_test, y_pred, average='macro') * 100
        f = f1_score(y_test, y_pred, average='macro') * 100
        # Print performance metrics
        text.delete('1.0', END)
        print("Accuracy: {:.2f}%".format(acc))
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", clr)
        text.insert(END,  " ML Model  Accuracy = " + str(acc) + "\n")
        text.insert(END,  " ML Model Precision = " + str(p) + "\n")
        text.insert(END,  " ML Model Recall = " + str(r) + "\n")
        text.insert(END,  " ML Model F1-Score = " + str(f) + "\n")
        class_names = ('sword','Sniper','SMG','shotgun','Handgun','Grenadelauncher','Bazooka','AutomaticRifle')
        sns.heatmap(cm, annot = True, cmap = "Blues", xticklabels = class_names, yticklabels = class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix of GNB Classifier")
        plt.show()
        
    else:
        # Data Loading & Preprocessing
        Categories = ['sword','Sniper','SMG','shotgun','Handgun','Grenadelauncher','Bazooka','AutomaticRifle']
        flat_data_arr = [] #input array
        target_arr = [] #output array
        datadir = r"data"
        #path which contains all the categories of images
        for i in Categories:
            print(f'loading... category : {i}')
            path = os.path.join(datadir,i)
            for img in os.listdir(path):
                img_array=imread(os.path.join(path,img))
                img_resized=resize(img_array,(150,150,3))  #resized to have a width of 150 pixels, a height of 150 pixels, and 3 color channels (r,g,b) helps to ensure that all images in your dataset have the same dimensions,
                flat_data_arr.append(img_resized.flatten()) #an image represented as a 2D or 3D array is converted into a 1D array.
                target_arr.append(Categories.index(i))
                print(f'loaded category:{i} successfully')
                flat_data=np.array(flat_data_arr)
                target=np.array(target_arr)
        np.save(os.path.join(datadir, 'flat_data.npy'), flat_data)
        np.save(os.path.join(datadir, 'target.npy'), target)
  

def cnnModel():
    global model,acc1,p1,r1,f1
    global accuracy,Y_test
    #text.delete('1.0', END)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        json_file.close()    
        model.load_weights("model/model_weights.h5")
        model._make_predict_function()   
        print(model.summary())
        f = open('model/history.pckl', 'rb')
        accuracy = pickle.load(f)
        f.close()
        acc1 = accuracy['accuracy']
        acc1 = acc1[9] * 100
        text.insert(END,"Deep Learning Model Accuracy is = "+str(acc1)+ "\n")
        y_pred1 = model.predict(X_test)
        y_pred1=np.argmax(y_pred1,axis=1)
        # Example: Reshape a singleton array
        Y_test = np.argmax(Y_test,axis=1)
        print(Y_test)
        cm = confusion_matrix(Y_test, y_pred1)
        clr = classification_report(Y_test, y_pred1, target_names=shapes)
        p1 = precision_score(Y_test, y_pred1, average='macro') * 100
        r1 = recall_score(Y_test, y_pred1, average='macro') * 100
        f1 = f1_score(Y_test, y_pred1, average='macro') * 100
        # Print performance metrics
        text.delete('1.0', END)
        print("Classification Report:\n", clr)
        text.insert(END,  " Deep Learning Model Precision = " + str(p1) + "\n")
        text.insert(END,  " Deep Learning Model Recall = " + str(r1) + "\n")
        text.insert(END,  " Deep Learning Model F1-Score = " + str(f1) + "\n")
        text.insert(END,  " CNN Model Confusion matrix ="+"\n" + str(cm) + "\n")
        class_names = ('sword','Sniper','SMG','shotgun','Handgun','Grenadelauncher','Bazooka','AutomaticRifle')
        sns.heatmap(cm, annot = True, cmap = "Blues", xticklabels = class_names, yticklabels = class_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix of CNN Classifier")
        plt.show()
        #Accuracy comparision graph
        acc1 = accuracy['accuracy']
        loss = accuracy['loss']
        plt.figure(figsize=(10,6))
        plt.grid(True)
        plt.xlabel('Iterations')
        plt.ylabel('Accuracy/Loss')
        plt.plot(acc1, 'ro-', color = 'green')
        plt.plot(loss, 'ro-', color = 'blue')
        plt.legend(['Accuracy', 'Loss'], loc='upper left')
        #plt.xticks(wordloss.index)
        plt.title('Performance Evaluation')
        plt.show()
        
    else:
        model = Sequential() #resnet transfer learning code here
        model.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim = 256, activation = 'relu'))
        model.add(Dense(output_dim = 8, activation = 'softmax'))
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        print(model.summary())
        hist = model.fit(X, Y, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
        model.save_weights('model/model_weights.h5')            
        model_json = model.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/history.pckl', 'rb')
        accuracy = pickle.load(f)
        f.close()
        acc = accuracy['accuracy']
        acc = acc[9] * 100
        text.insert(END,"CNN  Model Prediction Accuracy = "+str(acc))
        

def predict():
    global model
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    img = cv2.resize(img, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    test = np.asarray(im2arr)
    test = test.astype('float32')
    test = test/255
    preds = model.predict(test)
    predict = np.argmax(preds)
    img = cv2.imread(filename)
    img = cv2.resize(img, (500,500))
    cv2.putText(img, 'Image Classified as : '+shapes[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow('Image Classified as : '+shapes[predict], img)
    cv2.waitKey(0)
    
def graph():
    df = pd.DataFrame([
        ['GNB', 'Precision', p],
        ['GNB', 'Recall', r],
        ['GNB', 'F1 Score', f],
        ['GNB', 'Accuracy', acc],
        ['CNN', 'Precision', p1],
        ['CNN', 'Recall', r1],
        ['CNN', 'F1 Score', f1],
        ['CNN', 'Accuracy', acc1],
    ], columns=['Algorithms', 'Metrics', 'Value'])

    # Pivot the DataFrame and plot the graph
    pivot_df = df.pivot_table(index='Algorithms', columns='Metrics', values='Value', aggfunc='first')
    pivot_df.plot(kind='bar', figsize=(10, 6))

    # Set graph properties
    plt.title('Performance Metrics Comparison Between Algorithms')
    plt.ylabel('Score')
    plt.xlabel('Algorithms')
    plt.xticks(rotation=0)
    plt.tight_layout()
    # Display the graph
    plt.show()

    
def close():
    main.destroy()
    
    
font = ('times', 15, 'bold')
title = Label(main, text='weapon detection')
title.config(bg='powder blue', fg='olive drab')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

uploadButton = Button(main, text="Upload Dataset", command=uploadDataset)
uploadButton.place(x=20,y=100)
uploadButton.config(font=ff)


processButton = Button(main, text="Image Processing & Normalization", command=imageProcessing)
processButton.place(x=20,y=150)
processButton.config(font=ff)

mlpButton = Button(main, text="GNB Classifier", command=gnb)
mlpButton.place(x=20,y=200)
mlpButton.config(font=ff)

modelButton = Button(main, text="Build & Train CNN Model", command=cnnModel)
modelButton.place(x=20,y=250)
modelButton.config(font=ff)

predictButton = Button(main, text="Upload Test Image & Classify", command=predict)
predictButton.place(x=20,y=300)
predictButton.config(font=ff)

graphButton = Button(main, text="Performance Graph", command=graph)
graphButton.place(x=20,y=350)
graphButton.config(font=ff)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=20,y=400)
exitButton.config(font=ff)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=85)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)

main.config()
main.mainloop()
