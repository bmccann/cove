import requests
from argparse import ArgumentParser
import numpy as np


parser = ArgumentParser()
parser.add_argument('words', nargs='+', help='A list of words')
parser.add_argument('-o', '--output_file', help='where to serialize output')
args = parser.parse_args()

r = requests.post('http://localhost:8888/', json={'words': args.words})

# Returns concatenation of GloVe, CoVe, and Character n-gram embeddings
cove = np.array(r.json()['answer'])
np.save(args.output_file, cove)

print(cove)

