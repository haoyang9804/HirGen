from re import L
import subprocess
import os
import time
import signal
from tkinter import simpledialog
import spacy
# python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

def Red(string):
    return '\033[1;31m' + string + '\033[0m'

def Green(string):
    return '\033[1;32m' + string + '\033[0m'

segmFid = 0
_MAX_HOURS = 10
_CMD_RUN_TVMFUZZ = "python run.py"
# _CMD_RUN_TVMFUZZ_TEST = "TVM_BACKTRACE=1 python byproduct/program.py"
# _CMD_RUN_TVMFUZZ_MV = f"mv byproduct/program.py byproduct/program{segmFid}.py"

# inconsistency = 0

# def keepFileTVMfuzz():
#     global segmFid
#     p = subprocess.run(_CMD_RUN_TVMFUZZ_MV, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     segmFid += 1

f = open('result.txt', 'w')
f2 = open('result2.txt', 'w')
f3 = open('result3.txt', 'w')

def fun(p):
    str = p.stderr.decode('utf-8')
    strs = str.split('\n')
    keyline = ""
    for i in range(len(strs)):
        if strs[i].strip().startswith('0:'):
            keyline = strs[i+1]
    if keyline:
        eles = keyline.strip().split(',')
        filename = eles[0][6:-1].strip()
        lineNum = eles[1].strip()[5:].strip()
        return filename, lineNum
    return '', ''

bugid = 0
bugid2 = 0
bugid3 = 0

def startswithNum(line):
    line = line.strip()
    if line.startswith('0') or \
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

def recordBug():
    global bugid
    if not os.path.exists('./bugs'):
        os.mkdir('bugs')
    subprocess.run(f'mv byproduct/program.py bugs/program{bugid}.py', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    bugid += 1

def recordBug2():
    global bugid2
    if not os.path.exists('./bugs2'):
        os.mkdir('bugs2')
    subprocess.run(f'mv byproduct/program.py bugs2/program{bugid2}.py', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    bugid2 += 1

def recordBug3():
    global bugid3
    if not os.path.exists('./bugs3'):
        os.mkdir('bugs3')
    subprocess.run(f'mv byproduct/program.py bugs3/program{bugid3}.py', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    bugid3 += 1

visited = set()
visited2 = set()
visited3 = set()

def calculateSimWithSameLength(pureTxts1, pureTxts2):
    average = 0
    for i in range(len(pureTxts1)):
        p1 = nlp(pureTxts1[i])
        p2 = nlp(pureTxts2[i])
        average += p1.similarity(p2)
    return average / len(pureTxts1)

# if pureTxt not visited before, return true
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
            if average < 0.95:
                cnt += 1
    return cnt == len(visited)

def notvisited2(pureTxt): 
    global visited2
    cnt = 0
    for ele in visited2:
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
            if average < 0.95:
                cnt += 1
    return cnt == len(visited2)

def notvisited3(pureTxt): 
    global visited3
    cnt = 0
    for ele in visited3:
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
            if average < 0.95:
                cnt += 1
    return cnt == len(visited3)

beginTime = time.time()
while True:
    print(Red('======='))
    endTime = time.time()
    if endTime - beginTime > _MAX_HOURS * 3600:
        break
    
    p = subprocess.run(_CMD_RUN_TVMFUZZ, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1 = subprocess.run('./08env.sh', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p2 = subprocess.run('./expenv.sh', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if (not p1.returncode and p2.returncode):
        print('<returncode:', p1.returncode, p2.returncode)
        pureTxt = getPureReport(p2.stderr.decode('utf-8'))
        if notvisited(pureTxt):
            visited.add(pureTxt)
            recordBug()
    elif p1.returncode and p2.returncode:
        print('>returncode:', p1.returncode, p2.returncode)
        pureTxt1 = getPureReport(p1.stderr.decode('utf-8'))
        pureTxt2 = getPureReport(p2.stderr.decode('utf-8'))
        if ('Segmentation' in pureTxt2) and ('Segmentation' not in pureTxt1):
            if notvisited(pureTxt1):
                print('notvisited')
                visited.add(pureTxt1)
                recordBug()
        elif ('Segmentation' in pureTxt1) and ('Segmentation' not in pureTxt2):
            if notvisited2(pureTxt2):
                print('notvisited2')
                visited.add(pureTxt2)
                recordBug2()
        elif ('Segmentation' in pureTxt1) and ('Segmentation' in pureTxt2):
            recordBug() # manually check different segmentation fault
        else:
            if notvisited3(pureTxt1) and notvisited3(pureTxt2):
                print('notvisited3')
                pureTxts1 = pureTxt1.split('\n')
                pureTxts2 = pureTxt2.split('\n')
                if len(pureTxt1) != len(pureTxt2):
                    print('notvisited3 && not same length')
                    visited3.add(pureTxt1)
                    visited3.add(pureTxt2)
                    recordBug3()
                else:
                    sim = calculateSimWithSameLength(pureTxts1, pureTxts2)
                    if sim < 0.95: # different bug report
                        print('notvisited3 && not similar')
                        visited3.add(pureTxt1)
                        visited3.add(pureTxt2)
                        recordBug3()
            
                
    f.write(str(endTime-beginTime) + " " + str(bugid) + '\n')
    f.flush()
    f2.write(str(endTime-beginTime) + " " + str(bugid2) + '\n')
    f2.flush()
    f3.write(str(endTime-beginTime) + " " + str(bugid3) + '\n')
    f3.flush()
    print(endTime-beginTime, bugid, bugid2, bugid3)

f.close()
f2.close()
f3.close()