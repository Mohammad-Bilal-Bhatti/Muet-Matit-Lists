import csv

raw_data_file = 'First Marit List Raw.txt'
raw_file = open(raw_data_file, 'rt')


featured_data_file = 'Featured Data.csv'
featured_file = open(featured_data_file, 'wt', newline='')  # Open the file in write text mode.

csv_writer = csv.writer(featured_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

print("Executing...")
lines = raw_file.readlines()
for line in lines:
    tokens = line.split(" ")  # Split the line on the basics of white space
    last4columns = tokens[-4:]  # Get the last 4 Columns
    last4columns[-1] = last4columns[-1].strip()  # Remove /n | /t | white space
    csv_writer.writerow(last4columns)  # Write the last 4 Columns to the list.

featured_file.close()
print("Success!!!")
