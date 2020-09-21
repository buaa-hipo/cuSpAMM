import os

LoNum = 32
TESTTIME = 30
CHECK = 1

Norm = 0.4
T = 256
DEVICEDIM = 1
PART = 1
foname = 'out'
E = [1e2, 1e1, 1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

fo = open(foname, 'w')
fo.close()

for i in range(0, 1):
    T = 16384
    trun = 0
    for t in range(0, 100):
        #trun = E[t]
        trun = 0.0015*(t+1)+0.005
        f = open('para.h', 'w')
        f.write('#define LoNum %d\n' % LoNum)
        f.write('const float Norm=%f;\n' % Norm)
        f.write('#define TESTTIME %d\n' % TESTTIME)
        f.write('#define T %d\n' % T)
        f.write('#define DEVICEDIM %d\n' % DEVICEDIM)
        f.write('#define PART %d\n' % PART)
        f.write('#define CHECK %d\n' % CHECK)
        f.write('#define TRUNCATIONNUM %f\n' % trun)
        f.close()
        os.system("./run_spareseExp.sh>>"+foname)
