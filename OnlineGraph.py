import csv
import matplotlib.pyplot as plt
import numpy as np
import time

#this is for Paris 311.84
upper_bound_Paris = 311.84
upper_bound_AAU = 225.90
# Initialize empty lists to store the data
n = 5
data_model2 = [[] for i in range(n)]
data_model3 = [[] for i in range(n)]
data_model4 = [[] for i in range(n)]
data_model5A = [[] for i in range(n)]
data_model5B = [[] for i in range(n)]
data_Time = [[] for i in range(n)]

data_C  = [[] for i in range(n)]

abs_errors_model2 = [[] for i in range(n)]
abs_errors_model3 = [[] for i in range(n)]
abs_errors_model4 = [[] for i in range(n)]
abs_errors_model5A = [[] for i in range(n)]
abs_errors_model5B = [[] for i in range(n)]

ERROR_2 = np.array([],dtype=np.float64)
ERROR_3 = np.array([],dtype=np.float64)
ERROR_4 = np.array([],dtype=np.float64)
ERROR_5a = np.array([],dtype=np.float64)
ERROR_5b = np.array([],dtype=np.float64)

M3Time = np.array([],dtype=np.float64)
M4Time = np.array([],dtype=np.float64)
M5aTime  = np.array([],dtype=np.float64)
M5bTime  = np.array([],dtype=np.float64)

previous_row = []

# Define a function to update the plot
def update_plot(event):
    global previous_row 
    global ERROR_2 
    global ERROR_3 
    global ERROR_4 
    global ERROR_5a 
    global ERROR_5b
    global M3Time
    global M4Time
    global M5aTime 
    global M5bTime 
    

    # Load the latest rows from each file
    with open('Model3Predictions.csv', 'r') as file_2, open('Model3Predictions.csv', 'r') as file_3, open('Model4Predictions.csv', 'r') as file_4, open('Model5APredictions.csv', 'r') as file_5a, open('Model5BPredictions.csv', 'r') as file_5b, open('TimeToPredict.csv', 'r') as file_Time, open('C.csv', 'r') as file_C:
        # Skip header rows
        next(file_2)
        next(file_3)
        next(file_4)
        next(file_5a)
        next(file_5b)
        next(file_Time)
        

        next(file_C)

        # Initialize empty lists to store the latest rows from each file
        latest_row_2 = []
        latest_row_3 = []
        latest_row_4 = []
        latest_row_5a = []
        latest_row_5b = []
        latest_row_Time = []

        latest_row_C = []

        # Load the latest rows from each file
        for row in csv.reader(file_2):
            latest_row_2 = row
        for row in csv.reader(file_3):
            latest_row_3 = row
        for row in csv.reader(file_4):
            latest_row_4 = row
        for row in csv.reader(file_5a):
            latest_row_5a = row
        for row in csv.reader(file_5b):
            latest_row_5b = row
        for row in csv.reader(file_Time):
            latest_row_Time = row
        
        for row in csv.reader(file_C):
            latest_row_C = row

        # Extract the index values from the rows
        index_2 = int(latest_row_2[0])
        index_3 = int(latest_row_3[0])
        index_4 = int(latest_row_4[0])
        index_5a = int(latest_row_5a[0])
        index_5b = int(latest_row_5b[0])
        index_Time = int(latest_row_Time[0])

        index_C = int(latest_row_C[0])
        true_Y = float(latest_row_C[-1])
        #print("YYYYYYYYYYY : ",true_Y)
        #print("and index_C", true_Y)
        # Check if the index values match and the row is different from the previous one
        if index_2 == index_3 == index_4 == index_5a == index_5b == index_Time == index_C and latest_row_3 != previous_row and true_Y != 0 and true_Y < upper_bound_Paris:
            
            # Extract the relevant data from the rows
            data_2_new = list(map(float, latest_row_2[1:]))
            data_3_new = list(map(float, latest_row_3[1:]))
            data_4_new = list(map(float, latest_row_4[1:]))
            data_5a_new = list(map(float, latest_row_5a[1:]))
            data_5b_new = list(map(float, latest_row_5b[1:]))
            data_Time_new = list(map(float, latest_row_Time[1:]))

            data_C_new = list(map(float, latest_row_C[1:]))

            abs_error_model2 = np.abs(np.abs(np.array(data_2_new)) - np.abs(np.array(data_C_new)))
            abs_error_model3 = np.abs(np.abs(np.array(data_3_new)) - np.abs(np.array(data_C_new)))
            abs_error_model4 = np.abs(np.abs(np.array(data_4_new)) - np.abs(np.array(data_C_new)))
            abs_error_model5a = np.abs(np.abs(np.array(data_5a_new)) - np.abs(np.array(data_C_new)))
            abs_error_model5b = np.abs(np.abs(np.array(data_5b_new)) - np.abs(np.array(data_C_new)))
            
           
            for i in range(n):
                data_model2[i].append(data_2_new[i])
                data_model3[i].append(data_3_new[i])
                data_model4[i].append(data_4_new[i])
                data_model5A[i].append(data_5a_new[i])
                data_model5B[i].append(data_5b_new[i])
                data_Time[i].append(data_Time_new[i])

                

                data_C[i].append(data_C_new[i])

                abs_errors_model2[i].append(abs_error_model2[i])
                abs_errors_model3[i].append(abs_error_model3[i])
                abs_errors_model4[i].append(abs_error_model4[i])
                abs_errors_model5A[i].append(abs_error_model5a[i])
                abs_errors_model5B[i].append(abs_error_model5b[i])
            #print("len :",len(data_Time_new))  
            #print("Time : ",data_Time[0],data_Time[1],data_Time[2],data_Time[3],data_Time[4])
            
            M3Time = np.append(M3Time,data_Time_new[0])
            M4Time = np.append(M4Time,data_Time_new[7])
            M5aTime = np.append(M5aTime,data_Time_new[8])
            M5bTime = np.append(M5bTime,data_Time_new[9])
            #print("M3 :",np.mean(M3Time))
            #print("M4 :",np.mean(M4Time))
            #ERROR_3 = np.append(ERROR_3,mean_abs_error_model3)
            #ERROR_4 = np.append(ERROR_4,mean_abs_error_model4)
            #print("len of :",len(data_model2))
            # Clear the current figure
            plt.clf()

            # Create subplots for the data and absolute errors
            plt.subplot(1, 2, 1)
            #plt.plot(data_model2[4], 'b-', label='Model 2 Prediction')
            plt.plot(data_model3[4], 'g-', label='Model 3 Prediction')
            plt.plot(data_model4[4], 'c', label='Model 4 Prediction')
            plt.plot(data_model5A[4], 'm', label='Model 5A Prediction')
            plt.plot(data_model5B[4], 'y', label='Model 5B Prediction')
            
            #print("hej med dig ", len(data_model3[4]))
            plt.plot(data_C[4], 'r-', label='T_total')
            #print("the len of pt t: ",len(data_C[4]))
            plt.title(f" Model 3 t_DT: {np.mean(M3Time):.2f} ms,\nModel 4 t_DT: {np.mean(M4Time):.2f} ms,\nModel 5A t_DT: {np.mean(M5aTime):.2f} ms,\nModel 5B t_DT: {np.mean(M5bTime):.2f} ms,\nt_PT: {np.mean(data_C[4]):.2f} ms,\n\ny\u0302")
            #plt.title("y\u0302")
            plt.xlabel("Index")
            plt.ylabel("Time (ms)")
            plt.legend()
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
            plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
            # Plot all data points
            #plt.plot(range(len(data_model2[4])), data_model2[4], 'bo', alpha=0.3)
            
            plt.plot(range(len(data_model3[4])), data_model3[4], 'go', alpha=0.3)
            plt.plot(range(len(data_model4[4])), data_model4[4], 'co', alpha=0.3)
            plt.plot(range(len(data_model5A[4])), data_model5A[4], 'mo', alpha=0.3)
            plt.plot(range(len(data_model5B[4])), data_model5B[4], 'yo', alpha=0.3)
            

            plt.plot(range(len(data_C[4])), data_C[4], 'ro', alpha=0.3)
            """
            plt.subplot(1, 2, 2)
            plt.plot(abs_errors_AC[4], label='Error AC')
            plt.plot(abs_errors_BC[4], label='Error BC')
            plt.xlabel('Sample')
            plt.ylabel('Absolute Error')
            plt.title('Absolute Errors for Sample 4')
            plt.legend()
            plt.ylim(-10, 30) # Adjust the y-axis limits
            #plt.show()
            """
            plt.subplot(1, 2, 2)
            #plt.plot(abs_errors_model2[4], 'b-', label='Absolute Error M2', zorder=2)
            plt.plot(abs_errors_model3[4], 'g-', label='Absolute Error M3', zorder=2)
            plt.plot(abs_errors_model4[4], 'c', label='Absolute Error M4', zorder=2)
            plt.plot(abs_errors_model5A[4], 'm', label='Absolute Error M5A', zorder=2)
            plt.plot(abs_errors_model5B[4], 'y', label='Absolute Error M5B', zorder=2)
            

            plt.plot(range(len(data_model3[4])), abs_errors_model3[4], 'go', alpha=0.3)
            plt.plot(range(len(data_model3[4])), abs_errors_model4[4], 'co', alpha=0.3)
            plt.plot(range(len(data_model3[4])), abs_errors_model5A[4], 'mo', alpha=0.3)
            plt.plot(range(len(data_model3[4])), abs_errors_model5B[4], 'yo', alpha=0.3)

            #plt.scatter([0], [abs_errors_model2[4]], color='k', alpha=0.3, zorder=1)
            #plt.scatter([0], [abs_error_model3[4]], color='k', alpha=0.3, zorder=1)
            #plt.scatter([0], [abs_error_model4[4]], color='k', alpha=0.3, zorder=1)
            #plt.scatter([0], [abs_error_model5a[4]], color='k', alpha=0.3, zorder=1)
            #plt.scatter([0], [abs_error_model5b[4]], color='k', alpha=0.3, zorder=1)
            #plt.plot(abs_errors_model3[4], 'mo', alpha=0.3)
            #print(type(abs_error_AC[4]), type(abs_error_BC[4]))
            
            mean_abs_error_model2 = np.mean(abs_errors_model2[4])
            mean_abs_error_model3 = np.mean(abs_errors_model3[4])
            mean_abs_error_model4 = np.mean(abs_errors_model4[4])
            mean_abs_error_model5a = np.mean(abs_errors_model5A[4])
            mean_abs_error_model5b = np.mean(abs_errors_model5B[4])
            
            ERROR_2 = np.append(ERROR_2,mean_abs_error_model2)
            ERROR_3 = np.append(ERROR_3,mean_abs_error_model3)
            ERROR_4 = np.append(ERROR_4,mean_abs_error_model4)
            ERROR_5a = np.append(ERROR_5a,mean_abs_error_model5a)
            ERROR_5b = np.append(ERROR_5b,mean_abs_error_model5b)
            
            

            #plt.title(f"Absolute Errors. (MAE AC: {mean_abs_error_AC:.2f} ms, MAE BC: {mean_abs_error_BC:.2f} ms)")
            plt.title(f"MAE Model 3: {np.mean(ERROR_3):.2f} ms,\nMAE Model 4: {np.mean(ERROR_4):.2f} ms,\nMAE Model 5A: {np.mean(ERROR_5a):.2f} ms,\nMAE Model 5B: {np.mean(ERROR_5b):.2f} ms,\n\nAbsolute Errors")
            plt.ylabel("Time (ms)")
            plt.xlabel("Index")
            plt.legend()
            #plt.plot(abs_errors_model2[4], 'ko', alpha=0.3)
            

            
            # Add minor grid
            plt.minorticks_on()
            plt.grid(which='major', linestyle='-', linewidth='0.5', color='black')
            plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

                
               
            # Add title to the figure
            fig.suptitle("Models Performances", fontsize=16)
            # Update the plot
            plt.tight_layout()
            plt.draw()

            # Update

# Create the initial plot
fig = plt.figure(figsize=(12, 5))
fig.canvas.mpl_connect('motion_notify_event', update_plot)
update_plot(None)
#fig.title('Model 5b')

# Show the plot
plt.show()
