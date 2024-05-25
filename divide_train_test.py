import json
import os
import random

def load_data(types, total_subjects):
    all_data = []
    all_labels = []
    
    for sub_id in range(total_subjects):
        for type in types:
            pre = "00"
            if sub_id >= 100:
                pre = ""
            elif sub_id >= 10:
                pre = "0"
            for j in range(1):
                file_path = f'workspace/data/skeletons/{sub_id}/{pre}{sub_id}:90:{j}:0:{type}.json'
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_data.append(data)
                    all_labels.append(sub_id)  # Assuming label is sub_id, modify if needed
    
    return all_data, all_labels

def save_data(data, labels, train_data_path, test_data_path):
    train_size = int(len(data) * 0.8)
    
    train_data = data[:train_size]
    train_labels = labels[:train_size]
    
    test_data = data[train_size:]
    test_labels = labels[train_size:]
    
    with open(train_data_path, 'w') as f:
        json.dump({'data': train_data, 'labels': train_labels}, f)
    
    with open(test_data_path, 'w') as f:
        json.dump({'data': test_data, 'labels': test_labels}, f)

def main():
    types = ['nm', 'cl', 'bg', 'txt', 'ph', 'wss', 'wsf']
    total_subjects = 156
    
    data, labels = load_data(types, total_subjects)
    
    # Shuffle data with a seed for reproducibility
    random.seed(42)
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data[:], labels[:] = zip(*combined)
    
    # Save shuffled data into train and test files
    save_data(data, labels, 'train_data.json', 'test_data.json')

if __name__ == "__main__":
    main()