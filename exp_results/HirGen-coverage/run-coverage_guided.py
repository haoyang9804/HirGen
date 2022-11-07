import os
import time
import subprocess
os.chdir("build")
subprocess.run('cmake .. -G Ninja', shell=True)
subprocess.run('ninja', shell=True)
avgcov = 0;
timeT = 0.
for times in range(200):
  begtime = time.time()
  subprocess.run('  ./hirfuzz -coverage=yes', capture_output=True, shell=True)
  endtime = time.time()
  with open('score.txt') as f:
    lines = f.readlines()
    avgcov += int(lines[0])
  # print(avgcov, end=" ")
  timeT += endtime-begtime
  print(timeT, end=" ")
os.system('rm -rf cov.csv')