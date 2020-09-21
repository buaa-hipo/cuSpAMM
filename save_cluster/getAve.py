import os

LoNum = 32
TESTTIME = 1
CHECK = 1

Norm = 0
DEVICEDIM = 2
PART = 2
fo = open('getAve.out', 'w')
fo.close()

MADIM = [256, 512, 1024, 2048, 4096, 4096*2, 4096*4, 4096*8, 4096*16]

for i in range(0, 10):
    T = MADIM[i]
    f = open('para.h', 'w')
    f.write('#define LoNum %d\n' % LoNum)
    f.write('const float Norm=%f;\n' % Norm)
    f.write('#define TESTTIME %d\n' % TESTTIME)
    f.write('#define T %d\n' % T)
    f.write('#define DEVICEDIM %d\n' % DEVICEDIM)
    f.write('#define PART %d\n' % PART)
    f.write('#define CHECK %d\n' % CHECK)
    f.close()
    os.system("./getAve.sh>>getAve.out")
