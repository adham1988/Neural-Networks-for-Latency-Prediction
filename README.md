# Neural Networks for Latency Prediction
This README file provides an overview of the project and instructions on how to set it up and run it successfully. Please follow the steps below to ensure proper execution.

## Description 
This GitLab repository contains a system consisting of multiple scripts. The main script, server.py, listens for incoming TCP/UDP requests. The 'PT.py' script acts as a client, facilitating file downloads over TCP or UDP. The 'DT.py' script includes Neural Network models for predicting different stages of the file transmission process: connection time (t1), request and receive time (t2), storage time (t3), and decoding time (t4). The OnlineGraph.py script provides real-time visualization of actual and predicted latency, along with prediction time. Refer to 'modelstructures.png' for the Neural Network model architectures. This system enables efficient latency prediction and file transmission assessment.

## Requirements
To run this project, you need to have the following requirements installed:

TensorFlow 2.11.0
Please make sure you have installed TensorFlow version 2.11.0 or a compatible version before proceeding.

## Setup and Execution

If you plan to run the `server.py` script on a different machine than the one running the rest of the system, you need to make a small configuration change in the `PT.py` script.

By default, `PT.py` is configured to connect to the server using the IP address `localhost` (127.0.0.1), assuming that the server is running on the same machine. However, if you place the `server.py` script on a different machine, you'll need to modify the IP address in `PT.py` to match the IP address of the machine running the server.

**To run the system, please follow the steps below in the specified order:**
----------------------------------------------------------------------------

**DT.py**: Run the DT.py script and wait until all dependencies are successfully installed. You can monitor the console for any relevant messages indicating the completion of dependency installation. Once you see the message "The DT is ready to make predictions," it means the dependencies have been installed and the models are ready for prediction.

**server.py**: Run the server.py script. This script sets up the server component of the system, allowing communication between different modules.

**PT.py**: Run the PT.py script. This script executes the necessary processing tasks for the project. It requests the file to be downloaded from the server. 

**OnlineGraph.py**: Run the OnlineGraph.py script. This script handles the online graph functionality, enabling real-time visualization and analysis of the models' predictions, along with the time consumed for each model to complete the prediction.

Ensure that you run the scripts in the specified order to maintain the intended flow of the system.

Please note that you may need to provide additional input or configuration parameters as required by the scripts. Refer to the respective script files for any specific instructions or prompts.

## Troubleshooting
If you encounter any issues or errors during setup or execution, please consider the following troubleshooting steps:

Verify that you have installed TensorFlow 2.11.0 or a compatible version.
Check for any error messages or stack traces in the console output and refer to the relevant sections in the code or documentation for further guidance.
Ensure that all necessary dependencies are installed and configured correctly.
Double-check that the scripts are being run in the specified order as mentioned in this README.
If the problem persists, please consult the project documentation or seek assistance from the project maintainer or support team.

## Conclusion
You should now have the necessary information to set up and run the project successfully. If you have any further questions or require additional assistance, please don't hesitate to reach out.

**Note**: If you wish to rerun the system and reset it to its initial state, you can execute the `Empty.py` script. This script is responsible for resetting the system and allowing you to rerun it from scratch. To do so, follow the instructions provided in the "Rerunning the System" section above.

Thank you for using this project, and we hope it fulfills your requirements effectively.


## Authors and acknowledgment
We would like to express our appreciation to the following individuals for their contributions to this project:

- [Magnus Melgaard](https://gitlab.com/mistermelgaard) - Implemented the core functionality of the Neural Networks models.
- [Linette Anil](https://gitlab.com/linetteA) - Responsible for data preprocessing and training of the Neural Network models.

We would also like to thank the open-source community for their inspiration and the developers of the libraries and frameworks used in this project.

## License
This project is licensed under the MIT Thesis Project License. For more information, please see the [LICENSE.txt](LICENSE.txt) file.

