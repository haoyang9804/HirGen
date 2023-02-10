import os

files = os.listdir('1')
for file in files:
  path = os.path.join('1', file)
  print(path)