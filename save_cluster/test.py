import os
MADIM = [512, 1024, 2048, 4096, 4096*2, 4096*4, 4096*8]
E = [1e2, 1e1, 1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
LoNum = 32
TESTTIME = 20
CHECK = 1
DEVICEDIM = 2
PART = 1
ALG = 0
EXP = 1

outfile = 'paper_exp_FP32_MUL'
fo = open(outfile, 'w')
fo.close()
norm_file = open('normALG', 'r')

if ALG == 1:
    for T in MADIM:
        for i in range(0, 6):
            line = norm_file.readline()
            if line == '\n':
                line = norm_file.readline()
            for P in (1, 2):
                f = open('para.h', 'w')
                f.write('#define LoNum %d\n' % LoNum)
                f.write('const float Norm='+line.split("\n")[0]+';\n')
                f.write('#define TESTTIME %d\n' % TESTTIME)
                f.write('#define T %d\n' % T)
                f.write('#define DEVICEDIM %d\n' % DEVICEDIM)
                f.write('#define PART %d\n' % P)
                f.write('#define CHECK %d\n' % CHECK)
                f.write('#define MATRIXNOR %d\n' % 0)
                f.write('#define MATRIXALG %d\n' % ALG)
                f.write('#define MATRIXEXP %d\n' % EXP)
                f.close()
                os.system("./run.sh>>"+outfile)

if EXP == 1:
    for T in MADIM:
        for e in E:

            f = open('para.h', 'w')
            f.write('#define LoNum %d\n' % LoNum)
            f.write('const float Norm= '+str(e)+';\n')
            f.write('#define TESTTIME %d\n' % TESTTIME)
            f.write('#define T %d\n' % T)
            f.write('#define DEVICEDIM %d\n' % DEVICEDIM)
            f.write('#define PART %d\n' % PART)
            f.write('#define CHECK %d\n' % CHECK)
            f.write('#define MATRIXNOR %d\n' % 0)
            f.write('#define MATRIXALG %d\n' % ALG)
            f.write('#define MATRIXEXP %d\n' % EXP)
            f.close()
            os.system("./run.sh>>"+outfile)
