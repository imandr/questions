# coding: utf-8
from normalize import normalize_question
import pandas as pd
import sys

test_data = pd.read_csv(sys.argv[1])
test_data.fillna("", inplace=True)
questions = test_data["question1"].tolist()
questions += test_data["question2"].tolist()
normalized = map(normalize_question, questions)
words = set(" ".join(normalized).split())
print len(words)
