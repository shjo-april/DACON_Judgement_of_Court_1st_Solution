import json
import pandas as pd

def convert_csv2xlsx(domain='train'):
    data = []
    df = pd.read_csv(f"./open/{domain}.csv")

    for i in range(0, len(df)):
        data_id = df['ID'][i]
        users = [df['first_party'][i], df['second_party'][i]]
        text = df['facts'][i]

        for user_i in range(len(users)):
            if 'Title: \t ' in users[user_i]:
                users[user_i] = users[user_i].split('Title: \t ')[1]

        if domain == 'train':
            if int(df['first_party_winner'][i]) == 1: gt = 'Victory'
            else: gt = 'Defeat'
        else:
            gt = ''
        
        data.append(
            {
                'test_id': data_id,
                'The first party': users[0],
                'The second party': users[1], 
                'facts': text,
                'output': gt
            }
        )
    
    json.dump(data, open(f'./open/{domain}.json', 'w', encoding='utf-8'), indent='\t')

convert_csv2xlsx('train')
convert_csv2xlsx('test')

with open('./open/train.json', "r", encoding='utf-8') as file:
    data = json.load(file)

length = len(data) // 3
A_paths = data[:length]
B_paths = data[length:2*length]
C_paths = data[2*length:]

train_set, test_set = A_paths+C_paths, B_paths

with open(f'./open/trainval.json', 'w', encoding='utf-8') as file:
    json.dump({'train': train_set, 'validation': test_set}, file, indent='\t', ensure_ascii=False)