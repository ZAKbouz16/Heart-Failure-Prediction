#DatSet Source

#  https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/code?datasetId=1226038

#Import libraries

import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import QtCore 
import csv
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from sklearn.model_selection import  train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Run Plot generating code contained in the same folder,
# This will create a new folder and throw data represenation in it

path = './Data Represenation'
dir_list = os.listdir(path)
if len(dir_list)==0:
    os.system('python ./Plots_in_Jupyter.py')

cols=["Age","Sex","ChestPain","RestBP","Chol","Fbs","RestECG","MaxHR","ExAng","Oldpeak","Slope","Ca","Thal","AHD"]
df2=pd.read_csv('heart.csv',usecols=cols)

df=df2.copy()

# Drop rows with missing values
df.dropna(inplace=True)

# Encode categorical variables
cat_vars = ['ChestPain', 'Thal', 'AHD']   #label encoding == Yes or no.....1 and 0.
le = LabelEncoder()
df['ChestPain' ] = le.fit_transform(df['ChestPain'])
le2 = LabelEncoder()
df['Thal' ] = le2.fit_transform(df['Thal'])
le3 = LabelEncoder()
df['AHD' ] = le3.fit_transform(df['AHD'])

# Scale numerical variables
num_vars = ['Age', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca']
scaler = StandardScaler()
df[num_vars] = scaler.fit_transform(df[num_vars])

corr=df.corr().to_html()   # correlation to HTML
# Split the data into training and testing sets  80% and 20%
X_train, X_test, y_train, y_test = train_test_split(df.drop('AHD', axis=1), df['AHD'], test_size=0.2, random_state=42)


# Create the 4 models and train them.
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
            
from sklearn.naive_bayes import GaussianNB
            
nb = GaussianNB()
nb.fit(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
                
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Class to load PNG 

class MySubTabWidget(QTabWidget):
    def __init__(self, image_paths):
        super().__init__()
        self.setTabPosition(QTabWidget.North)

# add a layout to each tab and set the image as its widget
        for image in image_paths:
            tab1 = QWidget()
            layout1 = QVBoxLayout(tab1)
            image_label1 = QLabel()
            layout1.addWidget(image_label1)
            tab1.setLayout(layout1)
            self.addTab(tab1, (image.split(".")[0])[2:])
            image4 = QPixmap("Data Represenation/"+image)
            image_label1.setPixmap(image4)
            image_label1.setScaledContents(True)
 
# Our frame
class MyTabWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        """
        """
# create the tabs  
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tab5 = QWidget()
        self.tab6 = QWidget()
        self.tab7 = QWidget()
        
        
# Tab1
# add a label with an image to the first tab
        self.image_label = QLabel(self.tab1)
        self.image_label.setPixmap(QPixmap('Images/healthcare.jpg'))
        self.image_label.setScaledContents(True) 

       
# add a layout to the first tab and set the image label as its widget
        layout = QVBoxLayout(self.tab1)
        layout.addWidget(self.image_label)
        self.tab1.setLayout(layout)
        

#Tab 2 : Apps Infomation

        layout = QHBoxLayout(self.tab2)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(1)
        self.tree.setHeaderHidden(True)
        item1 = QTreeWidgetItem(self.tree)
        item1.setText(0, "Libraries used")
        item2 = QTreeWidgetItem(self.tree)
        item2.setText(0, "App Flow Chart")


        layout.addWidget(self.tree,0)
        self.text_edit2 = QTextEdit()
        self.text_edit2.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        layout.addWidget(self.text_edit2,3)
        self.tab2.setLayout(layout)

        
        
# Tab 3: Description of all our data set features.
        layout = QVBoxLayout(self.tab3)
        self.text_edit3 = QTextEdit()
        self.text_edit3.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)

        layout.addWidget(self.text_edit3,3)
        f = open('DataSet Description to HTML.txt','r').read()
        self.text_edit3.setHtml(f)
        self.tab3.setLayout(layout)
        
# Tab 4 import and display CVS File data:
        
# add labels to the other tabs

        self.table_view = QTableView(self.tab4)
        self.table_model = QStandardItemModel()
        self.table_view.setModel(self.table_model)
        
        with open('heart.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row_index, row_data in enumerate(reader):
                for column_index, column_data in enumerate(row_data):
                    item = QStandardItem(column_data)
                    self.table_model.setItem(row_index, column_index, item)
        
# add the table view to a layout and set this layout for the tab
        layout = QHBoxLayout(self.tab4)
        layout.addWidget(self.table_view)
        self.tab4.setLayout(layout)

# Tab 5 : Create SubTab and load all Plots genereated from Jupyter Code.

        self.tabbed_pane = QTabWidget(self.tab5)
        layout = QVBoxLayout(self.tab5) 
        lisofimage=[i for i in os.listdir('./Data Represenation/') if i.endswith(".png")]
        self.sub_tab1 = MySubTabWidget(lisofimage)
        layout.addWidget(self.sub_tab1)
        layout.addWidget(self.tabbed_pane)
        self.tab5.setLayout(layout)

        
# Tab 6 : Models training and testing interface
# Set subtab Model with drop down menu to show all models.
        layout = QHBoxLayout(self.tab6)
        self.tree2 = QTreeWidget()
        self.tree2.setColumnCount(1)
        self.tree2.setHeaderHidden(True)
        item1 = QTreeWidgetItem(self.tree2)
        item1.setText(0, "Model")
        
        item2 = QTreeWidgetItem(item1)
        item2.setText(0, "KNN")
        
        item3 = QTreeWidgetItem(item1)
        item3.setText(0, "Naive Bayse")
        
        item4 = QTreeWidgetItem(item1)
        item4.setText(0, "Decision Tree")

        item5 = QTreeWidgetItem(item1)
        item5.setText(0, "SVM")

        item6 = QTreeWidgetItem(item1)
        item6.setText(0, "random forest")

        layout.addWidget(self.tree2,1)
        self.text_edit = QTextEdit()
        self.text_edit.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse)
        layout.addWidget(self.text_edit,3)
        self.tab6.setLayout(layout)
        
# Tab 7 : New Patient Information and Predict Interface Tab

        layout = QGridLayout(self.tab7)


# Create the two labels and add them to the layout
        #layout left and right
        
        al=QtCore.Qt.AlignCenter

# Age new patient input tab
        labelAge = QLabel("Age")
        layout.addWidget(labelAge,0,2,al)
        self.lineEditAge = QLineEdit()
        self.lineEditAge.setFixedWidth(200)
        self.lineEditAge.setMaxLength(3)
        self.lineEditAge.setValidator(QIntValidator())
        layout.addWidget(self.lineEditAge,0,3)
        
# Sex new patient input tab
        layout.addWidget(QLabel("sexe"),2,0,al)
        self.comboBoxSex = QComboBox()
        self.comboBoxSex.setGeometry(QtCore.QRect(100, 70, 111, 22))
        self.comboBoxSex.addItem("--SELECT--")
        self.comboBoxSex.addItem("man")
        self.comboBoxSex.addItem("woman")
        layout.addWidget(self.comboBoxSex,2,1)
        
# Chest Pain new patient input tab
        layout.addWidget(QLabel("Chest Pain"),1,0,al)
        self.comboBoxChestPain = QComboBox()
        self.comboBoxChestPain.setGeometry(QtCore.QRect(100, 70, 111, 22))
        self.comboBoxChestPain.addItem("--SELECT--")  
        self.comboBoxChestPain.addItem("asymptomatic")
        self.comboBoxChestPain.addItem("nonanginal")
        self.comboBoxChestPain.addItem("typical")
        layout.addWidget(self.comboBoxChestPain,1,1)
        
# RestBP new patient input tab
        labelRestBP = QLabel("RestBP")
        layout.addWidget(labelRestBP,1,2,al)
        self.lineEditRestBP = QLineEdit()
        self.lineEditRestBP.setFixedWidth(200)
        self.lineEditRestBP.setMaxLength(3)
        self.lineEditRestBP.setValidator(QIntValidator())
        layout.addWidget(self.lineEditRestBP,1,3)

# Fasting Blood Sugar new patient input tab    
        labelFbs = QLabel("Fbs")
        layout.addWidget(labelFbs,2,2,al)
        self.lineEditFbs = QLineEdit()
        self.lineEditFbs.setFixedWidth(200)
        self.lineEditFbs.setMaxLength(3)
        self.lineEditFbs.setValidator(QIntValidator())
        layout.addWidget(self.lineEditFbs,2,3)
# Rest ECG new patient input tab
        labelRestECG= QLabel("RestECG")
        layout.addWidget(labelRestECG,6,0,al)
        self.lineEditRestECG = QLineEdit()
        self.lineEditRestECG.setFixedWidth(200)
        self.lineEditRestECG.setMaxLength(3)
        self.lineEditRestECG.setValidator(QIntValidator())
        layout.addWidget(self.lineEditRestECG,6,1)
#MaxHR new patient input tab
        labelMaxHR = QLabel("MaxHR")
        layout.addWidget(labelMaxHR,3,2,al)
        self.lineEditMaxHR = QLineEdit()
        self.lineEditMaxHR.setFixedWidth(200)
        self.lineEditMaxHR.setMaxLength(3)
        self.lineEditMaxHR.setValidator(QIntValidator())
        layout.addWidget(self.lineEditMaxHR,3,3)

#ExAng new patient input tab
        labelExAng= QLabel("ExAng")
        layout.addWidget(labelExAng,4,0,al)
        self.lineEditExAng = QLineEdit()
        self.lineEditExAng.setFixedWidth(200)
        self.lineEditExAng.setMaxLength(1)
        self.lineEditExAng.setValidator(QIntValidator())
        layout.addWidget(self.lineEditExAng,4,1)

#OldPeak new patient input tab
        labelOldpeak = QLabel("Oldpeak")
        layout.addWidget(labelOldpeak,4,2,al)
        self.lineEditOldpeak = QLineEdit()
        self.lineEditOldpeak.setFixedWidth(200)
        self.lineEditOldpeak.setMaxLength(2)
        self.lineEditOldpeak.setValidator(QIntValidator())
        layout.addWidget(self.lineEditOldpeak,4,3)

#Slope new patient input tab
        labelSlope= QLabel("Slope")
        layout.addWidget(labelSlope,5,0,al)
        self.lineEditSlope = QLineEdit()
        self.lineEditSlope.setFixedWidth(200)
        self.lineEditSlope.setMaxLength(2)
        self.lineEditSlope.setValidator(QIntValidator())
        layout.addWidget(self.lineEditSlope,5,1)

#CA new patient input tab
        labelCa = QLabel("Ca")
        layout.addWidget(labelCa,5,2,al)
        self.lineEditCa = QLineEdit()
        self.lineEditCa.setFixedWidth(200)
        self.lineEditCa.setMaxLength(2)
        self.lineEditCa.setValidator(QIntValidator())
        layout.addWidget(self.lineEditCa,5,3)

#thal new patient input tab
        layout.addWidget(QLabel("Thal"),3,0,al)
        self.comboBoxThal = QComboBox()
        self.comboBoxThal.addItem("--SELECT--")
        self.comboBoxThal.addItem("fixed")
        self.comboBoxThal.addItem("normal")
        #self.comboBoxThal.addItem("nontypical")
        self.comboBoxThal.addItem("reversable")
        layout.addWidget(self.comboBoxThal,3,1)
        
#Serum Cholesterol new patient input tab        
        labelChol = QLabel("Chol")
        layout.addWidget(labelChol,6,2,al)
        self.lineEditChol = QLineEdit()
        self.lineEditChol.setFixedWidth(200)
        self.lineEditChol.setMaxLength(3)
        self.lineEditChol.setValidator(QIntValidator())
        layout.addWidget(self.lineEditChol,6,3)

#Model Choice input tab
        layout.addWidget(QLabel("Model"),0,0,al)
        self.comboBoxMODEL = QComboBox()
        self.comboBoxMODEL.addItem("--SELECT--")
        self.comboBoxMODEL.addItem("KNN")
        self.comboBoxMODEL.addItem("Naive Bays")
        self.comboBoxMODEL.addItem("Decision Tree")
        self.comboBoxMODEL.addItem("SVM")
        self.comboBoxMODEL.addItem("random forest")
# The click to predict       
        layout.addWidget(self.comboBoxMODEL,0,1) 
        l=QVBoxLayout()
        button=QPushButton("Predict")
        button.setGeometry(QtCore.QRect(100, 70, 111, 22))
        button.clicked.connect(self.submit_clicked)        
        layout.addWidget(QLabel(""),7,2)
        layout.addWidget(button,7,2)     
        self.tab6.setLayout(layout)





# Tabs Clicking.
        self.tree.itemClicked.connect(self.handle_item_clicked)
        self.tree2.itemClicked.connect(self.handle_item_clicked2)
        # add the tabs to the tab widget
        self.addTab(self.tab1, 'Home')
        self.addTab(self.tab2, 'App Tutorial')
        self.addTab(self.tab3, 'Dataset Description') 
        self.addTab(self.tab4, 'Dataset')
        self.addTab(self.tab5, 'DataSet Graph Representation')
        self.addTab(self.tab6, 'Models')
        self.addTab(self.tab7, 'Predict')
        
#Function to predict patien case.
    def submit_clicked(self):
        
        data={'Age': [None],'Sex': [None],'ChestPain': [None],
              'RestBP': [None],'Chol': [None],'Fbs': [None],
              'RestECG': [None],'MaxHR': [None],'ExAng': [None],
              'Oldpeak': [None],'Slope': [None],'Ca': [None],
              'Thal': [None]
        }
# Input all data
        try:
            age=self.lineEditAge.text()
            age=int(age)
            data['Age']=[age]
            restBP=self.lineEditRestBP.text()
            restBP=int(restBP)
            data['RestBP']=[restBP]
            chol=self.lineEditChol.text()
            chol=int(chol)
            data['Chol']=[chol]
            fbs=self.lineEditFbs.text()
            fbs=int(fbs)
            data['Fbs']=fbs
            restECG=self.lineEditRestECG.text()
            restECG=int(restECG)
            data['RestECG']=[restECG]
            maxHR=self.lineEditMaxHR.text()
            maxHR=int(maxHR)
            data['MaxHR']=[maxHR]
            exAng=self.lineEditExAng.text()
            exAng=int(exAng)
            data['ExAng']=[exAng]
            oldpeak=self.lineEditOldpeak.text()
            oldpeak=float(oldpeak)
            data['Oldpeak']=[oldpeak]
            slope=self.lineEditSlope.text()
            slope=int(slope)
            data['Slope']=[slope]
            ca=self.lineEditCa.text()
            ca=float(ca)
            data['Ca']=[ca]
       
            chestPain=self.comboBoxChestPain.currentText()
            if chestPain!="--SELECT--":
                data['ChestPain']=[chestPain]
            thal=self.comboBoxThal.currentText()
            if thal!="--SELECT--":
                data['Thal']=[thal]
            sex=self.comboBoxSex.currentText() 
            if sex!="--SELECT--":
                data['Sex']=['woman','man'].index(sex)
        
            data['ChestPain']=le.transform(data['ChestPain'])
            data['Thal']=le2.transform(data['Thal'])
            d=pd.DataFrame(data)
            num_vars = ['Age', 'RestBP', 'Chol', 'Fbs', 'RestECG', 'MaxHR', 'ExAng', 'Oldpeak', 'Slope', 'Ca']
            
            d[num_vars] = scaler.transform(d[num_vars])
            model=self.comboBoxMODEL.currentText()
            models={"Decision Tree":dt,
                    "Naive Bays":nb,
                    "KNN":knn,
                    "SVM":svc,
                    "random forest":rf
            }
            if model!="--SELECT--":
                model=models[model]
               
                predict=model.predict(d)

# Pop uo message for result of the prediction

                if predict[0] == 1:
                    msg_good = QMessageBox()
                    msg_good.setWindowTitle('Resutls')
                    msg_good.setIcon(QMessageBox.Information)
                    msg_good.setWindowIcon(QIcon('Images/heart_healthy.png'))
                    msg_good.setText('You are healthy.')
                    msg_good.setStyleSheet("QMessageBox {min-width: 230px; min-height: 20px; font-size: 100px; background : lightgreen;}")
                                                                                          
                    msg_good.exec_()
                    
                elif predict[0] == 0:
                    msg_bad = QMessageBox()
                    msg_bad.setWindowTitle('Resutls')
                    msg_bad.setIcon(QMessageBox.Critical)
                    msg_bad.setWindowIcon(QIcon('Images/heart-rate-monitor.png'))
                    msg_bad.setText('You \'re at risk.')
                    msg_bad.setStyleSheet("QMessageBox {min-width: 230px; min-height: 20px; font-size: 100px; background : red;}")
                    msg_bad.exec_()
                
# Pop up message in case you missed to choose the model               
            else:
                msg_bad = QMessageBox()
                msg_bad.setWindowTitle('Model')
                msg_bad.setInformativeText('You didn\'t choose model')
                msg_bad.setIcon(QMessageBox.Critical)   
                msg_bad.exec_()
                
# Missing data Error Message
        except Exception as e:
            print(e)
            msg_bad = QMessageBox()
            msg_bad.setWindowTitle('Something is missing')
            msg_bad.setInformativeText('check if filled all fields')
            msg_bad.setIcon(QMessageBox.Critical)         
            msg_bad.exec_()


# Click to Fill Info about models on Tab2
    def handle_item_clicked(self, item, column):
        model=item.text(column)
        if (model=="App Flow Chart"):
            f="<br><img src='./Images/Flowchart.png'/>"
            self.text_edit2.setHtml(f)

        elif (model=="Libraries used"):
            f="<br><img src='./Images/Library.png'/>"
            self.text_edit2.setHtml(f)
         

# Click to Fill Info about models on Tab6
    def handle_item_clicked2(self, item, column):
        model=item.text(column)
        if(model=="Model"):
            text= "\nThis data set has  {} line and {} features<br>".format(df2.shape[0],df2.shape[1])
            text+='\nheader  <br>{}'.format(df2.head().to_html())
            text+="\nnumber of missing data   {} <br>".format(df2.isnull().sum().sum())
            text+="\nnumber of missing data   {} <br>".format(df2[['Ca','Thal']].isnull().sum().to_dict())
            text+='\nremoving missing data + Normalization <br>'
            text+='\nnew header   <br>{}'.format(df.head().to_html())
            text+="\nnumber of missing data   {} <br>".format(df.isnull().sum().sum())
            text+="\ncorrelation matrix <br> {} <br>".format(corr)
            text+="\nno correlation<br>"
            text+="\ndata are splitted {}% for train and {}% for test".format(80,20)
            text+="<br><img src='./Images/corr.png'/>"
            self.text_edit.setHtml(text)
        elif model=="KNN":
            text="definiton of knn<br>"
            text+=""
            k_values = range(1, 11)

# Train and evaluate the model for each K value
            best_k = None
            best_accuracy = 0
            for k in k_values:
                # Train the KNN model
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                
# Evaluate the KNN model
                y_pred = knn.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
# Choose the best K value
                if accuracy > best_accuracy:
                    best_k = k
                    best_accuracy = accuracy
            text+=f"Best K: {best_k}, Accuracy: {round(best_accuracy*100,2)} <br>"
            y_pred = knn.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            report=pd.DataFrame(report)
            text+="classification report <br>{}<br>".format(report.to_html())
            text+="confusion matrix <br>{}<br>".format(pd.DataFrame(confusion_matrix(y_test, y_pred)).to_html())
            self.text_edit.setHtml(text)

# Evaluate the Naive Bayse model

        elif model=="Naive Bayse":
            text="definition de naive bayse<br>"
            y_pred = nb.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            report=pd.DataFrame(report)
            text+="accuracy {}% <br>".format(round(accuracy_score(y_test, y_pred)*100,2))
            text+="classification report <br>{}<br>".format(report.to_html())
            text+="confusion matrix <br>{}<br>".format(pd.DataFrame(confusion_matrix(y_test, y_pred)).to_html())
            self.text_edit.setHtml(text)
            
# Evaluate the Decision Tree model            
        elif model=='Decision Tree':
            #DecisionTreeClassifier
            text="definition de DecisionTreeClassifier<br>"
            
            y_pred = dt.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            report=pd.DataFrame(report)
            text+="accuracy {}% <br>".format(round(accuracy_score(y_test, y_pred)*100,2))
            text+="classification report <br>{}<br>".format(report.to_html())
            text+="confusion matrix <br>{}<br>".format(pd.DataFrame(confusion_matrix(y_test, y_pred)).to_html())
            self.text_edit.setHtml(text)
            
# Evaluate the SVM model:              
        elif model=='SVM':
            text="definition de SVM<br>"
            
            y_pred = svc.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            report=pd.DataFrame(report)
            text+="accuracy {}% <br>".format(round(accuracy_score(y_test, y_pred)*100,2))
            text+="classification report <br>{}<br>".format(report.to_html())
            text+="confusion matrix <br>{}<br>".format(pd.DataFrame(confusion_matrix(y_test, y_pred)).to_html())
            self.text_edit.setHtml(text)
            
# Evaluate the random forest model:             
        elif model=='random forest':
            text="definition de random forest<br>"
            y_pred = rf.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            report=pd.DataFrame(report)
            text+="accuracy {}% <br>".format(round(accuracy_score(y_test, y_pred)*100,2))
            text+="classification report <br>{}<br>".format(report.to_html())
            text+="confusion matrix <br>{}<br>".format(pd.DataFrame(confusion_matrix(y_test, y_pred)).to_html())
            self.text_edit.setHtml(text)

class MyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Heart Failure Prediction')
        self.tab_widget = MyTabWidget()
        self.setCentralWidget(self.tab_widget)
        self.resize(1200, 900)
        self.setMinimumSize(QtCore.QSize(1300, 700))
        self.setWindowIcon(QIcon('Images/heart_healthy.png'))
        self.show()

    def resizeEvent(self, event):
        self.tab_widget.tab1.layout().setGeometry(self.tab_widget.tab1.rect())
if __name__ == '__main__':
    app = QApplication([])
    window = MyMainWindow()
    window.setWindowIcon(QIcon('Images/heart_healthy.png'))
    
    window.show()
    app.exec_()
