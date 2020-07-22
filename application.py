from request import start
from model_generate import generate
from classifier import classify

print('Generate model...')
generate()
print('Downloading twitter images....')
start()
print('Execute classifier...')
result = classify()
print(f'The CR7 appear {result}% in analyzed images!')
