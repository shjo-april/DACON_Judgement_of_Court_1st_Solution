import tqdm
import json
import argparse
import numpy as np

from strsimpy.jaro_winkler import JaroWinkler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default='./llm/dacon_submissions/test_qa_albert-xxlarge-v2_vicuna-13b-v1.3.json')
    args = parser.parse_args()
    
    f = open(f'./submissions/230709_final_results.csv', 'w')
    f.write('ID,first_party_winner\n')

    final_win_rate = []
    test_data = json.load(open(args.json, 'r', encoding='utf-8'))

    jarowinkler = JaroWinkler()
    
    lines = open('./submissions/230709_classification_models.csv', 'r').readlines()[1:]
    ensemble_dict = {}
    for line in lines:
        ensemble_id, ensemble = line.strip().split(',')
        ensemble_dict[ensemble_id] = int(ensemble)

    for data in tqdm.tqdm(test_data):
        test_id = data['id']

        first = data['The first party']
        second = data['The second party']
        prediction = data['prediction']

        outputs = prediction.split(' is ')
        if len(outputs) == 2:
            prediction = outputs[1].replace('\"', '').replace('.', '')

            sim1 = jarowinkler.similarity(prediction, first)
            sim2 = jarowinkler.similarity(prediction, second)

            if sim1 >= 0.95:
                first_party_winner = 0
            elif sim2 >= 0.95:
                first_party_winner = 1
            else:
                first_party_winner = ensemble_dict[test_id]
        else:
            first_party_winner = ensemble_dict[test_id]

        final_win_rate.append(first_party_winner)
        f.write(f'{test_id},{first_party_winner}\n')

    f.close()
    