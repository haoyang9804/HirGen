import os
import subprocess
bugfiles = os.listdir('bugs')
f = open('recordbuginfo.txt', 'w')
f2 = open('uniquebugs.txt', 'w')
buginfo = set()
for bf in bugfiles:
  p = subprocess.run('TVM_BACKTRACE=1 python bugs/' + bf, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  errtext = p.stderr.decode('utf-8')
  preline = ""
  f.write(f'----------{bf}----------\n')
  for line in errtext.split('\n'):
    if 'TVMError' in line:
      f.write(preline.strip() + '\n')
      if preline.strip() not in buginfo:
        buginfo.add(preline.strip())
        f2.write(f'{bf}\n')
    elif 'Check failed' in line:
      f.write(line.strip() + '\n')
    preline = line
    f.flush()
    f2.flush()
f.close()  
f2.close()