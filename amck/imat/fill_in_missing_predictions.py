import csv
import random


with open('predictions.csv') as predictions_csv:
    predictions = dict()
    for line in predictions_csv:
        line = line.strip()
        if line == 'id,predicted':
            continue
        id, category = line.split(sep=',')
        id = int(id)
        category = int(category)
        predictions[id] = category

    correction_mapping = [-1] + sorted([x for x in range(1, 128 + 1)], key=lambda x: str(x))

    ids = predictions.keys()
    for i in range(1, 12800 + 1):
        if i not in ids:
            predictions[i] = random.randint(1, 128 + 1)
        else:
            predictions[i] = correction_mapping[predictions[i]]

    with open('predictions_filled_in.csv', 'w') as predictions_filled_in_csv:
        csv_writer = csv.writer(predictions_filled_in_csv, quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(['id', 'predicted'])
        csv_writer.writerows(predictions.items())
