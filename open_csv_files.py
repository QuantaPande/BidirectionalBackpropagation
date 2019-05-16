import numpy as np

# Author : Tanmayan Pande
# EE 559

# This code is a general purpose code, which reads a CSV file and transfers the
# content to a numpy.ndarray. It is assumed that the data is unlabelled and is
# in numerical format. The function open_csv is the only function in this module
# which does the job of opening the file given in the filename variable. The 
# default delimiter is the comma, while the default end_string is the \n 
# character


def open_csv(filename, delimiter = ',', end_string = '\n'):
    
    #Dummy csv_list variable to store all the values in the CSV file
    csv_list = [];
    file_object = open(filename, "r")

    # The file_object is the file pointer to the CSV file. The read() function 
    # reads the file ans stores the data in csv_string in the form of a string.
    # csv_list then store this value in a list for easy access
    csv_string = file_object.read()
    csv_list = list(csv_string)
    i = 0;
    count = 0;\

    # The function then counts the number of delimiter characters in one row of
    # the CSV file. This value is then corrected to include the last column and
    # used as the number of columns in the numpy.ndarray, of type float
    while(csv_list[i] != end_string):
        if(csv_list[i] == delimiter):
            count = count + 1;
        i = i + 1;
    count = count + 1
    csv_array = np.empty([1, count], np.float);
    j = 0;

    # The variable last_break stores the location of the last line break or end
    # string character in the file. The for loop runs through the file looking 
    # for end string characters. 
    last_break = 0;
    k = 0;
    for i in range(0, len(csv_list)):
        if csv_list[i] == end_string:
            # Once it finds the end character, the loop then checks the delimiter 
            # charater. Anything between two delimter characters is treated as a 
            # value. The variable start_charac stores the first character of the 
            # new row, or rather the last character of the previous row.
            start_charac = last_break;
            n = 0
            for j in range(last_break, i+1):
                # Check if the the current character is the delimiter or
                # the end of string character. If yes, join all characters
                # between start_charac and the current characters. Append
                # this to the dummy_array variable to convert to the 
                # numpy.ndarray data type. Convert the value to the float
                # data type. Increment the index variables
                if(csv_list[j] == delimiter or csv_list[j] == end_string):
                    dummy_array = np.asarray(''.join(csv_list[start_charac:j]))
                    csv_array[k, n] = dummy_array.astype(np.float)
                    n = n + 1;

                    #Update the start_charac variable to the variable to the index 
                    # after the current delimiter character
                    start_charac = j + 1;

            # Update the last_break variable to point to the next line      
            last_break = i + 1;

            # Update the row index variable for the csv_array variable
            k = k + 1;

            # Append a new row to the csv_array variable
            csv_array = np.append(csv_array, [np.zeros(count)], axis = 0)

    # Delete the last line of the csv_array variable, as it was unnecessarily
    # added to the array at the end of the final iteration of the for loop
    csv_array = np.delete(csv_array, k, 0)

    # Close the file and return the csv_array variable
    file_object.close()
    return csv_array;
            
            
    