
#Neural Network Project 

#Import Modules
#####################################################

import numpy as np
import matplotlib.pyplot as plt
import csv

 
#Import dataset from CVS file
#####################################################
data = []

with open("crimes.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        #variable generator
        data.append(row)
        
train_inputs = []        
train_outputs = []
learn = 0.1
num = 0
sex = []
location = []
time = []

#filter to gather 1000 admissible training data points
####################################################
while len(train_inputs) < 1000:
    num += 1
    
    #ensure no blanks are used
    if ((data[num][6] != str()) and (data[num][13] != str()) and (data[num][12] != str()) and (data[num][0] != str()) and (data[num][13] != 'U')):
        
        #input data
        #select_in = ["""(1/46)*"""float(data[num][3]),"""(1/24)*"""float(data[num][6])]
        select_in = [float(data[num][3]),float(data[num][6]),float(data[num][4])]
        location.append(float(data[num][3])) 
        time.append(float(data[num][6]))
        
        train_inputs.append(select_in)
    
        if data[num][13] == 'M':
            sex.append(0.0)
            select_out = [float(data[num][12]),0.0]
        elif data[num][13] == 'F':
            sex.append(1.0)
            select_out = [float(data[num][12]),1.0]
        
        train_outputs.append(select_out)

    else:
        continue
    
#Plot sex data for observation   
plt.scatter(time,location,s = 200,c = sex, alpha = 1)
plt.title('Offender Sex (Blue = M, Red = F)',fontsize=18,color='r')
plt.xlabel('Time of Day',fontsize=14,color='b')
plt.ylabel('Location',fontsize=14,color='g')
plt.legend()
plt.grid()
plt.show() 

    
#Activation Function and weights
########################################################    
 
def sigmoid(x):  
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):  
    return x *(1-x)

#Initialize weights
np.random.seed(1)
weights1 = 2 * np.random.random((3,2))  #input layer to hidden neuron weights
weights2 = 2 * np.random.random((2,1))  #hidden layer to output neuron weights
print("Starting Weights:")
print('\n',weights1,'\n','\n',weights2)

t = 0
err = []
hid_err = [0,0]
c1 = 0.01 #normalized conversion 1
c2 = 0.00001 #normalized conversion 2
adjustments2 = np.array([[0.0],
                         [0.0]])
                         
adjustments1 = np.array([[0.0,0.0],
                         [0.0,0.0],
                         [0.0,0.0]])

while t < 1:
    t += 1
    output = []
    for i in range(0,len(train_inputs)-100):

        #input layer 
        x = np.array([(train_inputs[i][0])*(c1),(train_inputs[i][1])*(c1),(train_inputs[i][2])*(c2)])
        
        #(calculate hidden layer from input layer)
        hidden_layer = sigmoid(np.dot(x,weights1))
   
        #(calculate output layer from hidden layer)
        output_layer = sigmoid(np.dot(hidden_layer,weights2)) 
        output.append(output_layer)

        y = (np.array(float(train_outputs[i][0])))*c1
        error = np.array([y-output_layer])
        squared_error = (sum(error)**2)/2
    
        
        #Now back proprogate from output weight to hidden weight
        delta_out = sigmoid_derivative(output_layer)*error
        
        #adjustments for weights between hidden and output layers  (hidden to input)     
        for a in range(2):
            for b in range(1):
                adjustments2[a][b] = learn*delta_out[b]*hidden_layer[a]
                
        #hidden layer node errors
        hid_err[0] = hidden_layer[0] * (1 - hidden_layer[0]) * (weights2[0][0] * delta_out[0])
        hid_err[1] = hidden_layer[1] * (1 - hidden_layer[1]) * (weights2[1][0] * delta_out[0])
        
        weights2 += adjustments2
        
        delta_in = np.array([hid_err[0],hid_err[1]])
        
        #adjustments for weights between input and output layers  (hidden to outpout)
        for c in range(3):
            for d in range(2):
                adjustments1[c][d] = learn*delta_in[d]*x[c]
                
        weights1 += adjustments1
        
        err.append(squared_error)
        


plt.plot(err)
plt.title('Age Error Tracker',fontsize=18,color='r')
plt.xlabel('Learning Sequence',fontsize=14,color='b')
plt.ylabel('Error',fontsize=14,color='g')
plt.grid()
plt.show()

##########################################################
# TEST THE AGES 
##########################################################
tt = 0
squared_error_test = []
while tt < 1:
    tt += 1
    output = []
    for i in range(len(train_inputs)-100,1000):

        #input layer 
        xx = np.array([(train_inputs[i][0])*c1,(train_inputs[i][1])*c1,(train_inputs[i][1])*c2])
        hl = sigmoid(np.dot(xx,weights1)) 
        ol = sigmoid(np.dot(hl,weights2))
        yy = np.array(float(train_outputs[i][0]))*c1
        error_test = [yy-ol]
        sqt = ((error_test[0])**2)/2
        squared_error_test.append(sqt)
      
        
print("Max Squared Error Percent:",max(squared_error_test)*100,"%")
avg = sum(squared_error_test)/len(squared_error_test)
print("Average Error:",avg[0]*100,'%')
print('\n',"Adjusted Weights:")
print('\n',weights1,'\n','\n',weights2)

plt.plot(squared_error_test)
plt.title('Age Test Error Evaluation',fontsize=18,color='r')
plt.xlabel('Learning Sequence',fontsize=14,color='b')
plt.ylabel('Error',fontsize=14,color='g')
plt.grid()
plt.show()


#####################################################
#Part 2: Gender
#####################################################

adjustments2 = np.array([[0.0],
                         [0.0]])
                         
adjustments1 = np.array([[0.0,0.0],
                         [0.0,0.0],
                         [0.0,0.0]])
tg = 0
er = []
learn = 0.1
       
while tg < 1:
    tg += 1
    output = []
    for i in range(0,len(train_inputs)-100):

        #input layer 
        x = np.array([(train_inputs[i][0]),(train_inputs[i][1]),(train_inputs[i][2])])
        
        #(calculate hidden layer from input layer)
        hidden_layer = sigmoid(np.dot(x,weights1))
   
        #(calculate output layer from hidden layer)
        output_layer = (np.dot(hidden_layer,weights2)) 
        
        output.append(output_layer)

        if (output_layer - 0.99) > 0:
            output_layer = 1

        else:
            output_layer = 0

        y = (np.array(float(train_outputs[i][1])))#*c1
        error = np.array([y-output_layer])
        squ_error = (sum(error)**2)/2

        
        #Now back proprogate from output weight to hidden weight
        delta_out = learn*(output_layer)*error
        
        #adjustments for weights between hidden and output layers  (hidden to input)     
        for a in range(2):
            for b in range(1):
                adjustments2[a][b] = learn*delta_out[b]*hidden_layer[a]
                
        #hidden layer node errors
        hid_err[0] = hidden_layer[0] * (1 - hidden_layer[0]) * (weights2[0][0] * delta_out[0])
        hid_err[1] = hidden_layer[1] * (1 - hidden_layer[1]) * (weights2[1][0] * delta_out[0])
        
        weights2 += adjustments2
        
        delta_in = np.array([hid_err[0],hid_err[1]])
        
        #adjustments for weights between input and output layers  (hidden to outpout)
        for c in range(3):
            for d in range(2):
                adjustments1[c][d] = learn*delta_in[d]*x[c]
                
        weights1 += adjustments1
        er.append(error)

plt.plot(er)
plt.axis([60, 900, -0.2, 1.2])
plt.title('Gender Error Tracker',fontsize=18,color='r')
plt.xlabel('Learning Sequence',fontsize=14,color='b')
plt.ylabel('Error',fontsize=14,color='g')

plt.grid()
plt.show()

##########################################################
# TEST THE GENDERS 
##########################################################
tt = 0 
incorrect = []   
threshold = 0.2 
count = 0
squared_error_test = []
while tt < 1:
    tt += 1
    output = []
    for i in range(len(train_inputs)-100,1000):

        #input layer 
        xx = np.array([(train_inputs[i][0])*c1,(train_inputs[i][1])*c1,(train_inputs[i][1])*c2])
        hl = sigmoid(np.dot(xx,weights1)) 
        ol = sigmoid(np.dot(hl,weights2))
        yy = np.array(float(train_outputs[i][1]))
        error_test2 = [yy-ol]
        sqt2 = ((error_test2[0])**2)/2
        squared_error_test.append(sqt2)
        if sqt2 >= threshold:
            incorrect.append(1)

correct_percent = (1-(sum(incorrect))/len(squared_error_test))*100
print("Gender Percent Correctly Guessed:",'\n',correct_percent,'%','\n')            
        
print("Max Squared Error Value:",max(squared_error_test)*100,'%')
avg = sum(squared_error_test)/len(squared_error_test)
print("Average Error:",avg[0]*100,'%')

plt.plot(squared_error_test)
plt.title('Gender Test Error Evaluation',fontsize=18,color='r')
plt.xlabel('Learning Sequence',fontsize=14,color='b')
plt.ylabel('Error',fontsize=14,color='g')
plt.grid()
plt.show()
