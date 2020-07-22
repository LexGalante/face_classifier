from face_classifier.request import start
from face_classifier.model_generate import model_generate
from face_classifier.classifier import classify

print('Generate model...')
model_generate()
print('Downloading twitter images....')
start()
print('Execute classifier...')
result = classify()
print(f'The CR7 appear {result}% in analyzed image!')
