import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow import keras
import tensorflow as tf
import time
import csv
import random
import os

# parameters for unstandirisation 
T1_std = 15.8529
T1_mean = 10.9362
T1_min = 0
T1_max = 176.9971999892732

T2_std = 45.9239
T2_mean = 60.5327
T2_min = 2.511
T2_max = 227.142

T3_std = 4.9899
T3_mean = 7.57097
T3_min = 0.25529
T3_max =  70.2520

T4_std = 1.36451
T4_mean = 1.42012
T4_min = 0.4011
T4_max = 47.9598

T5_std = 53.32449237
T5_mean = 80.5982016
T5_min = 8.6714000
T5_max = 240.41969

t_1 = np.array([],dtype=np.float64)
t_2 = np.array([],dtype=np.float64)
t_3 = np.array([],dtype=np.float64)
t_4 = np.array([],dtype=np.float64)
t_5A = np.array([],dtype=np.float64)
t_5B = np.array([],dtype=np.float64)
Indexs = np.array([]).astype(int)
rows = []



"""
Convert Keras Model to TensorFlow Lite Model and Perform Inference

for each model we convert a Keras models (`NN`) to a TensorFlow Lite model and using an interpreter to perform inference.

Steps:
1. Load the Keras model.
2. Save the model in the TensorFlow SavedModel format.
3. Create a TensorFlow Lite converter from the saved model.
4. Configure the converter to specify supported operations and allow custom ops.
5. Convert the model to the TensorFlow Lite format.
6. Create a TensorFlow Lite interpreter from the serialized model.
7. Allocate tensors required by the interpreter.
8. Retrieve input and output tensor details.

Usage:
- Replace 'NN1' with the path to your Keras model file.
- Run the code to obtain the TensorFlow Lite model and interpreter for inference.
"""


Model2 = keras.models.load_model('Model2NN')
export_dir = 'saved_model/Model2'
tf.saved_model.save(Model2, export_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
converter.allow_custom_ops = True
converter.experimental_new_converter = True
tflite_model_M2 = converter.convert()
# Load TFLite model and allocate tensors.
interpreter_M2 = tf.lite.Interpreter(model_content=tflite_model_M2)
interpreter_M2.allocate_tensors()
    # Get input and output tensors.
input_details_M2 = interpreter_M2.get_input_details()
output_details_M2 = interpreter_M2.get_output_details()



Model3 = keras.models.load_model('Model3NN')
export_dir = 'saved_model/Model3'
tf.saved_model.save(Model3, export_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
converter.allow_custom_ops = True
converter.experimental_new_converter = True
tflite_model_M3 = converter.convert()
# Load TFLite model and allocate tensors.
interpreter_M3 = tf.lite.Interpreter(model_content=tflite_model_M3)
interpreter_M3.allocate_tensors()
    # Get input and output tensors.
input_details_M3 = interpreter_M3.get_input_details()
output_details_M3 = interpreter_M3.get_output_details()


NN1 = keras.models.load_model('NN1')
export_dir = 'saved_model/NN1'
tf.saved_model.save(NN1, export_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
converter.allow_custom_ops = True
converter.experimental_new_converter = True
tflite_NN1 = converter.convert()
# Load TFLite model and allocate tensors.
interpreter_NN1 = tf.lite.Interpreter(model_content=tflite_NN1)
interpreter_NN1.allocate_tensors()
    # Get input and output tensors.
input_details_NN1 = interpreter_NN1.get_input_details()
output_details_NN1 = interpreter_NN1.get_output_details()



NN2 = keras.models.load_model('NN2')
export_dir = 'saved_model/NN2'
tf.saved_model.save(NN2, export_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
converter.allow_custom_ops = True
converter.experimental_new_converter = True
tflite_NN2 = converter.convert()
# Load TFLite model and allocate tensors.
interpreter_NN2 = tf.lite.Interpreter(model_content=tflite_NN2)
interpreter_NN2.allocate_tensors()
    # Get input and output tensors.
input_details_NN2 = interpreter_NN2.get_input_details()
output_details_NN2 = interpreter_NN2.get_output_details()



NN3 = keras.models.load_model('NN3')
export_dir = 'saved_model/NN3'
tf.saved_model.save(NN3, export_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
converter.allow_custom_ops = True
converter.experimental_new_converter = True
tflite_NN3 = converter.convert()
# Load TFLite model and allocate tensors.
interpreter_NN3 = tf.lite.Interpreter(model_content=tflite_NN3)
interpreter_NN3.allocate_tensors()
    # Get input and output tensors.
input_details_NN3 = interpreter_NN3.get_input_details()
output_details_NN3 = interpreter_NN3.get_output_details()



NN4 = keras.models.load_model('NN4')
export_dir = 'saved_model/NN4'
tf.saved_model.save(NN4, export_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
converter.allow_custom_ops = True
converter.experimental_new_converter = True
tflite_NN4 = converter.convert()
# Load TFLite model and allocate tensors.
interpreter_NN4 = tf.lite.Interpreter(model_content=tflite_NN4)
interpreter_NN4.allocate_tensors()
    # Get input and output tensors.
input_details_NN4 = interpreter_NN4.get_input_details()
output_details_NN4 = interpreter_NN4.get_output_details()



NN5A = keras.models.load_model('NN5A')
export_dir = 'saved_model/NN5A'
tf.saved_model.save(NN5A, export_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
converter.allow_custom_ops = True
converter.experimental_new_converter = True
tflite_NN5A = converter.convert()
# Load TFLite model and allocate tensors.
interpreter_NN5A = tf.lite.Interpreter(model_content=tflite_NN5A)
interpreter_NN5A.allocate_tensors()
    # Get input and output tensors.
input_details_NN5A = interpreter_NN5A.get_input_details()
output_details_NN5A = interpreter_NN5A.get_output_details()



NN5B = keras.models.load_model('NN5B')
export_dir = 'saved_model/NN5B'
tf.saved_model.save(NN5B, export_dir)
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
converter.allow_custom_ops = True
converter.experimental_new_converter = True
tflite_NN5B = converter.convert()
# Load TFLite model and allocate tensors.
interpreter_NN5B = tf.lite.Interpreter(model_content=tflite_NN5B)
interpreter_NN5B.allocate_tensors()
    # Get input and output tensors.
input_details_NN5B = interpreter_NN5B.get_input_details()
output_details_NN5B = interpreter_NN5B.get_output_details()


input_ModelNN2 = np.array([-0.6898548530552769,-1.1342133389223963,-1.0994348264717169,0.05201854794336662])
input_ModelNN3 = np.array([0,0,1,1,0.20220479480420456,1.0,2.362685265911072,1.49797,0.7041203919738684,1.19039])

input_NN1 = np.array([1,0,0,1,0])
input_NN2 = np.array([0,0,1,1,0.20220479480420456,1.0,2.362685265911072,1.49797,0.7041203919738684,1.19039])
input_NN3 = np.array([1.19039,0])
input_NN4 = np.array([1])
input_NN5A = np.array([1,0,1,1])
input_NN5B = np.array([1,0,1,1])

Data = np.genfromtxt("database.txt", dtype=float,
                     encoding=None, delimiter=",")    


# reshape array 
input_ModelNN2 = input_ModelNN2.reshape(1, -1)
input_ModelNN3 = input_ModelNN3.reshape(1, -1)
# expand the dimensions of inputs along the first axis.
input_NN2 = np.expand_dims(input_NN2, axis=0)
input_NN3 = np.expand_dims(input_NN3, axis=0)
input_NN4 = np.expand_dims(input_NN4, axis=0)
input_NN5A = np.expand_dims(input_NN5A, axis=0)
input_NN5B = np.expand_dims(input_NN5B, axis=0)



# Model 2 a
# sets the input tensor of the TensorFlow Lite interpreter
interpreter_M2.set_tensor(input_details_M2[0]['index'], input_ModelNN2)
# invokes the TensorFlow Lite interpreter 
#  Invoking the interpreter runs the inference on the provided input data.
# It processes the input tensor, performs computations using the pre-trained model, and produces output tensors.
interpreter_M2.invoke()
# retrieve the output tensor from the TensorFlow Lite interpreter 
predictionM2 = interpreter_M2.get_tensor(output_details_M2[0]['index'])

# Model 3
interpreter_M3.set_tensor(input_details_M3[0]['index'], input_ModelNN3)
interpreter_M3.invoke()
predictionM3 = interpreter_M3.get_tensor(output_details_M3[0]['index'])

# NN1
interpreter_NN1.set_tensor(input_details_NN1[0]['index'], [input_NN1.astype(np.uint8)])
interpreter_NN1.invoke()
prediction1 = interpreter_NN1.get_tensor(output_details_NN1[0]['index'])

# NN2
interpreter_NN2.set_tensor(input_details_NN2[0]['index'], input_NN2)
interpreter_NN2.invoke()
prediction2 = interpreter_NN2.get_tensor(output_details_NN2[0]['index'])

# NN3
interpreter_NN3.set_tensor(input_details_NN3[0]['index'], input_NN3)
interpreter_NN3.invoke()
prediction3 = interpreter_NN3.get_tensor(output_details_NN3[0]['index'])

# NN4
interpreter_NN4.set_tensor(input_details_NN4[0]['index'], input_NN4.astype(np.float64))
interpreter_NN4.invoke()
prediction4 = interpreter_NN4.get_tensor(output_details_NN4[0]['index'])



outM2 = predictionM2[0][0]
outM3 = predictionM3[0][0]
out1 = prediction1[0][0]
out2 = prediction2[0][0]
out3 = prediction3[0][0]
out4 = prediction4[0][0]



arr2 = np.array([out1,out2,out3,out4])



arr2 = np.expand_dims(arr2, axis=0)

count = len(Data)

print("\n\n\t the DT is ready to make predictions")
print("\n\n")
while True:
    Data = np.genfromtxt("database.txt", dtype=float,
                     encoding=None, delimiter=",") 

    
                
    if count != len(Data):
        
        #print(count)
        inputs = Data[-1]
        # indexing the iteration
        Indexs = np.append(Indexs, len(t_2)+1) 
       
        # get inputs for Model 3
        input_ModelNN3 = np.array([inputs])

        # starte timing to get the time elapsed
        starttime = time.perf_counter()

        # Model 3 predicting 
        interpreter_M3.set_tensor(input_details_M3[0]['index'], input_ModelNN3)
        interpreter_M3.invoke()
        predictionM3 = interpreter_M3.get_tensor(output_details_M3[0]['index'])
   
        # register the time to predict for model 3
        timeM3 = time.perf_counter() - starttime
        
        # save the output 
        outM3 = predictionM3[0][0]
       
        # unstandarise to get it in ms
        y_hat_model3 = ((outM3*T5_std)+T5_mean)
        
        # save the Model 3 predictions
        with open('Model3Predictions.csv', mode='a', newline='') as f:
            # create a csv writer object
            writer = csv.writer(f)
            # write the data row
            writer.writerow([Indexs[-1],300,300,300,300,y_hat_model3])

        # get inputs for NN1
        input_NN1 = np.array([int(inputs[0]),int(inputs[1]),int(inputs[2]),int(inputs[3]),int(inputs[4])])

        # get inputs for NN2
        input_NN2 = np.array([inputs])
        
        # get inputs for NN3
        input_NN3 = np.array([inputs[7],int(inputs[5])])
        
        # get inputs for NN4
        input_NN4 = np.array([inputs[7]])
        
        # starte timing to get the time elapsed
        starttime2 = time.perf_counter()

        # NN1 predicting 
        interpreter_NN1.set_tensor(input_details_NN1[0]['index'], [input_NN1.astype(np.uint8)])
        interpreter_NN1.invoke()
        T_hat_1 = interpreter_NN1.get_tensor(output_details_NN1[0]['index'])
        
        # register the time to predict for NN1
        timeNN1 = time.perf_counter() - starttime2 
       

        # starte timing to get the time elapsed
        starttime3 = time.perf_counter()

        # NN2 predicting 
        interpreter_NN2.set_tensor(input_details_NN2[0]['index'], input_NN2)
        interpreter_NN2.invoke()
        T_hat_2 = interpreter_NN2.get_tensor(output_details_NN2[0]['index'])

        # register the time to predict for NN1
        timeNN2 = time.perf_counter() - starttime3 
        
        # starte timing to get the time elapsed
        starttime4 = time.perf_counter()

        # NN3 predicting 
        interpreter_NN3.set_tensor(input_details_NN3[0]['index'], [input_NN3])
        interpreter_NN3.invoke()
        T_hat_3 = interpreter_NN3.get_tensor(output_details_NN3[0]['index'])

        # register the time to predict for NN3
        timeNN3 = time.perf_counter() - starttime4 
        
        # starte timing to get the time elapsed
        starttime5 = time.perf_counter()

        # NN4 predicting
        interpreter_NN4.set_tensor(input_details_NN4[0]['index'], [input_NN4.astype(np.float64)])
        interpreter_NN4.invoke()
        T_hat_4 = interpreter_NN4.get_tensor(output_details_NN4[0]['index'])

        # register the time to predict for NN4
        timeNN4 = time.perf_counter() - starttime5 
        
        # save the outputs 
        out1 = T_hat_1[0][0]
        out2 = T_hat_2[0][0]
        out3 = T_hat_3[0][0]
        out4 = T_hat_4[0][0]
        
        # group the outputs into one array
        arr2 = np.array([out1,out2,out3,out4])
        
        # starte timing to get the time elapsed
        starttime5a = time.perf_counter()

        # NN5a predicting
        interpreter_NN5A.set_tensor(input_details_NN5A[0]['index'], [arr2.astype(np.float64)])
        interpreter_NN5A.invoke()
        prediction5A = interpreter_NN5A.get_tensor(output_details_NN5A[0]['index'])
        timeNN5a = time.perf_counter() - starttime5a 
        
        # starte timing to get the time elapsed
        starttime5b = time.perf_counter()

        # NN5b predicting
        interpreter_NN5B.set_tensor(input_details_NN5B[0]['index'], [arr2.astype(np.float32)])
        interpreter_NN5B.invoke()
        prediction5B = interpreter_NN5B.get_tensor(output_details_NN5B[0]['index'])

        # register the time to predict for NN3
        timeNN5b = time.perf_counter() - starttime5b 
       
        
        y_hat_A = prediction5A[0][0]
        y_hat_B = prediction5B[0][0]

       
        
        t_1 = np.append(t_1,((out1*T1_std)+T1_mean))
        t_2 = np.append(t_2,((out2*T2_std)+T2_mean))
        t_3 = np.append(t_3,((out3*T3_std)+T3_mean))
        t_4 = np.append(t_4,((out4*T4_std)+T4_mean))
        t_5A = np.append(t_5A,((y_hat_A*T5_std)+T5_mean))
        t_5B = np.append(t_5B,((y_hat_B*T5_std)+T5_mean))
        count = len(Data) 
        
        y_hat_model4 = t_1[-1] + t_2[-1] + t_3[-1] + t_4[-1] 
        
        # save the predicted values of each model to the corresponding file 
        with open('Model4Predictions.csv', mode='a', newline='') as f:
            # create a csv writer object
            writer = csv.writer(f)
            # write the data row
            writer.writerow([Indexs[-1],400,400,400,400,y_hat_model4])

        with open('Model5APredictions.csv', mode='a', newline='') as f:
            # create a csv writer object
            writer = csv.writer(f)
            # write the data row
            writer.writerow([Indexs[-1],t_1[-1],t_2[-1],t_3[-1],t_4[-1],t_5A[-1]])
        
        with open('Model5BPredictions.csv', mode='a', newline='') as f:
            # create a csv writer object
            writer = csv.writer(f)
            # write the data row
            writer.writerow([Indexs[-1],t_1[-1],t_2[-1],t_3[-1],t_4[-1],t_5B[-1]])
        

        M4_T = timeNN1 + timeNN2 + timeNN3 +timeNN4
        M5a_T = timeNN1 + timeNN2 + timeNN3 +timeNN4 + timeNN5a
        M5b_T = timeNN1 + timeNN2 + timeNN3 +timeNN4 +timeNN5b

        # save the time to predict for each model 
        with open('TimeToPredict.csv', mode='a', newline='') as f:
            # create a csv writer object
            writer = csv.writer(f)
            # write the data row
            writer.writerow([Indexs[-1],float(timeM3*1000),timeNN1*1000,timeNN2*1000,timeNN3*1000,timeNN4*1000,timeNN5a*1000,timeNN5b*1000,M4_T*1000,M5a_T*1000,M5b_T*1000])
        
        Data2 = np.genfromtxt("TrueDelays.txt", dtype=float,
                     encoding=None, delimiter=",") 

       



        
        
        

        
    time.sleep(0.33)








