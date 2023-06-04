import csv

# Function to empty a CSV file
def empty_csv(file_path):
    with open(file_path, 'w', newline='') as csvfile:
        csvfile.truncate()

# File paths for CSV files A and B
file_a = 'Model2Predictions.csv'
file_b = 'Model3Predictions.csv'
file_c = 'Model4Predictions.csv'
file_d = 'Model5APredictions.csv'
file_e = 'Model5BPredictions.csv'
file_f = 'TimeToPredict.csv'
file_g = 'C.csv'
# Empty CSV file A
empty_csv(file_a)
empty_csv(file_b)
empty_csv(file_c)
empty_csv(file_d)
empty_csv(file_e)
empty_csv(file_f)
empty_csv(file_g)

print("the files are empty")


# Read the first line of the file
with open("database.txt", "r") as file:
    first_line = file.readline()

# Write only the first line back to the file
with open("database.txt", "w") as file:
    file.write(first_line)