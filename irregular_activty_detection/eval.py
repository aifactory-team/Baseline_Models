import sys
import json

submit_file = sys.argv[2]
answer_file = sys.argv[1]

with open(answer_file, "r") as json_file:
    answer = json.load(json_file)

with open(submit_file, "r") as json_file:
    submit = json.load(json_file)

incorrect_clips = [k for k in answer if submit[k] != answer[k]]
print('{}'.format(1 - len(incorrect_clips) / len(answer)))
