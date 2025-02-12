import os
import json
from collections import defaultdict

def read_json(file_path):
    """Read a JSON file and return its content as a dictionary."""
    with open(file_path, 'r') as f:
        return json.load(f)

def average_metrics(json_files):
    """Calculate the average of all metrics found in the 'Scores' field for each dataset."""
    averages = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'count': 0}))
    
    for file in json_files:
        data = read_json(file)
        for partial_result in data:
            print(partial_result)
            dataset = partial_result['Dataset']
            scores = partial_result['Scores']
        
            # Dynamically process all metrics in 'Scores'
            for metric, value in scores.items():
                averages[dataset][metric]['total'] += value
                averages[dataset][metric]['count'] += 1
    
    # Calculate averages for each metric
    results = {}
    for dataset, metrics in averages.items():
        results[dataset] = {
            metric: values['total'] / values['count']
            for metric, values in metrics.items()
        }
    
    return results

def get_json_files(directory):
    """Recursively find all JSON files in the given directory."""
    json_files = []
    print(os.walk(directory))
    for root, dirs, files in os.walk(directory):
        print(files)
        for file in files:
            print(file)
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def main():
    # Define the directory containing the JSON files
    directory = 'results/Basic/'  # Change this to the correct path
    json_files = get_json_files(directory)
    averages = average_metrics(json_files)
    # print(json_files)
    
    # Print the results
    for dataset, metrics in averages.items():
        print(f"{dataset}: averages")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")
        print()  # Add a blank line for readability

if __name__ == "__main__":
    main()