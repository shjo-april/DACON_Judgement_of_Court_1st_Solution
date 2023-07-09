import numpy as np

# 1. classification
# path1 = 'submissions/230627_008_2.csv'
# path2 = 'submissions/230709_classification_models.csv'

# 2. classification + vicuna-13b-v1.3
path1 = 'submissions/230630_009_5.csv'
path2 = 'submissions/230709_final_results.csv'

first_lines = open(path1, 'r').readlines()[1:]
second_lines = open(path2, 'r').readlines()[1:]

corrects = []

for first, second in zip(first_lines, second_lines):
    first = first.strip()
    second = second.strip()

    correct = int(first.split(',')[1]) == int(second.split(',')[1])
    corrects.append(correct)

print('공식 결과와 {}% 일치합니다.'.format(int(np.mean(corrects)*100)))
