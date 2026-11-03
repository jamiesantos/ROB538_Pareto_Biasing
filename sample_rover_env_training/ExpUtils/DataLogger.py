import csv
import copy

class DataLogger:
    def __init__(self, data_fields: list, target_filename=None):
        '''
        data_fields (list): the order in which the fields appear here is the order in which they
        will be saved in the destination file.
        '''
        self.target_filename = target_filename
        # Save the data fields
        self.data_fields = copy.deepcopy(data_fields)
        # Create a dict of the data fields with empty values
        self.data = {}
        for data_field in self.data_fields:
            self.data[data_field] = None
        # Open and clear the file
        with open(self.target_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.data_fields)
            writer.writeheader()
    
    def add_data(self, key=None, value=None):
        '''
        Add data to be saved for this generation.
        WILL OVERWRITE THE PREVIOUS DATA HELD HERE.
        '''
        if key in self.data:
            self.data[key] = value
    
    def write_data(self):
        '''
        Will write the saved data into the file.
        '''
        if self.target_filename:
            with open(self.target_filename, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.data_fields)
                writer.writerow(self.data)
