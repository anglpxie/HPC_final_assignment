import os
import pandas as pd

class Parser:    
    def __init__(self, folder_to_load):
        self.folder_to_load = folder_to_load
        self.data = self.load_all()
    
    def parse_filename(self, filename):
        parts = filename.split('-')
        operation = parts[0]
        processes = parts[1][2:]
        algorithm = parts[2][1:].split('.')[0]
        if operation == 'bcast':
            algorithms = [
                "ignore",
                "basic_linear",
                "chain",
                "pipeline",
                "split_binary_tree",
                "binary_tree"
            ]
            algorithm = algorithms[int(algorithm)] # convert the algorithm number to name
        else:
            algorithms = [
                "ignore",
                "linear",
                "chain",
                "pipeline",
                "binary"
            ]
            algorithm = algorithms[int(algorithm)]
        return operation, processes, algorithm
    
    # loads a single .csv file and returns a pandas DataFrame
    def load(self, filepath):
        with open(filepath, 'r') as file:
            file_content = file.read()

            data_lines = []
            lines = file_content.strip().split('\n')

            for line in lines:
                stripped_line = line.strip()
                if stripped_line and not stripped_line.startswith('#'):
                    data_lines.append(stripped_line)

            df = pd.DataFrame(
                [line.split() for line in data_lines],
                columns=[
                    'Size',
                    'Avg_Latency',
                    'Min_Latency',
                    'Max_Latency',
                    'Iterations'
                ]
            )

            df = df.astype({'Size': int, 'Avg_Latency': float, 'Min_Latency': float, 'Max_Latency': float, 'Iterations': int})
            
            return df
    
    # loads all .csv files in the folder_to_load directory, merges them into a single DataFrame and returns it
    def load_all(self):
        all_data = []
        
        for root, dirs, files in os.walk(self.folder_to_load):
            for file in files:
                if file.endswith('.csv'):
                    filepath = os.path.join(root, file)
                    df = self.load(filepath)
                    operation, processes, algorithm = self.parse_filename(file)
                    df['Operation'] = operation
                    df['Processes'] = processes
                    df['Processes'] = df['Processes'].astype(int)
                    df['Algorithm'] = algorithm
                    all_data.append(df)
        
        return pd.concat(all_data, ignore_index=True)
    
    # saves the merged DataFrame to a .csv file
    def save_to_csv(self, output_file):
        self.data.to_csv(output_file, index=False)

parser = Parser('benchmarks')
parser.save_to_csv('benchmarks.csv')