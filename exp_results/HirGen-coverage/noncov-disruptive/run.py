import os
import subprocess
import time
import random
import signal
import spacy
nlp = spacy.load('en_core_web_sm')
def Red(string):
    return '\033[1;31m' + string + '\033[0m'

def Green(string):
    return '\033[1;32m' + string + '\033[0m'

MAX_HOURS = 24
MAX_GRAPHS = 200
ftime = open('time.txt', 'w')
# fid = 0
if not os.path.exists('bugs'):
    subprocess.run('mkdir -p bugs', shell=True)
# if not os.path.exists('build'):
#     subprocess.run('mkdir -p build', shell=True)
os.chdir("build")
# subprocess.run('cmake .. -G Ninja', shell=True)
# subprocess.run('ninja', shell=True)
bugid = 0
visited = set()

recordfile = open('record.txt', 'w')

def recordBug(timing, seg=False):
    global bugid
    if not os.path.exists('./bugs'):
        os.mkdir('bugs')
    if seg:
        p = subprocess.run(f'mv output.py ../bugs/outputseg{bugid}.py', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print('>>>', p.returncode)
    else:
        p = subprocess.run(f'mv output.py ../bugs/output{bugid}.py', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print('<<<', p.returncode)
    ftime.write(str(timing) + ": " + str(bugid) + '\n')
    bugid += 1


def notvisited(pureTxt):
    global visited
    cnt = 0
    for ele in visited:
        eles = ele.split('\n')
        pureTxts1 = pureTxt.split('\n')
        if len(eles) != len(pureTxts1):
            cnt += 1
        else:
            average = 0
            for i in range(len(eles)):
                e = nlp(eles[i])
                p = nlp(pureTxts1[i])
                average += e.similarity(p)
            average /= len(eles)
            print('average = ',average)
            if average < 0.50:
                cnt += 1
    return cnt == len(visited)

def startswithNum(line):
    line = line.strip()
    if  line.startswith('0') or \
            line.startswith('1') or \
                line.startswith('2') or \
                    line.startswith('3') or \
                        line.startswith('4') or \
                            line.startswith('5') or \
                                line.startswith('6') or \
                                    line.startswith('7') or \
                                        line.startswith('8') or \
                                            line.startswith('9'):
                                                return True
    return False

def getPureReport(txt):
    txt.strip()
    pureTxt = ''
    lines = txt.split('\n')
    for line in lines:
        if startswithNum(line):
            pureTxt += line.strip() + '\n'
    return pureTxt

cLevel = "relaxed"
begintime = time.time()
records = set()
while True:
    print(Red('======='))
    endtime = time.time()
    if endtime - begintime > MAX_HOURS * 3600:
        break

    subprocess.run('./hirfuzz -testing=df -coverage=no -clevel=relaxed', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p = subprocess.run('TVM_BACKTRACE=1 python output.py', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode:
        if 'Segmentation fault' in p.stderr.decode('utf-8'):
            print('Segmentation fault')
            with open('output.py', 'a') as tmpF:
                tmpF.write('# Segmentation Fault')
            recordBug(endtime-begintime, True)
            with open('cov.csv') as f:
                lastline = f.readlines()[-1]
                try:
                    op, dtype = lastline.split('@')[-1].split(';')
                    print(op, dtype)
                    if (op, dtype[:-1]) not in records:
                        recordfile.write(op + " " + dtype[:-1] + " " + str(bugid) + " " + str(endtime-begintime) + '\n')
                        records.add((op, dtype[:-1]))
                except:
                    print("except")
        else:
            if cLevel == "strict":
                txt = p.stderr.decode('utf-8')
                pureTxt = getPureReport(txt)
                if notvisited(pureTxt):
                    visited.add(pureTxt)
                    with open('output.py', 'a') as tmpF:
                        tmpF.write('\n\'\'\'')
                        tmpF.write(pureTxt)
                        tmpF.write('\n\'\'\'')
                    recordBug(endtime-begintime)
    print(str(endtime-begintime) + " " + str(bugid) + " " + str(len(visited)))
    ftime.flush()
    recordfile.flush()
ftime.close()
recordfile.close()
os.system('rm -rf cov.csv')