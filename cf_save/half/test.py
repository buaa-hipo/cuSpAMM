import os
# MADIM = [1024, 2048, 4096, 4096*2, 4096*4, 4096*8]
MADIM = [2048]
E = [1e2, 1e1, 1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
LoNum = 32
TESTTIME = 20
ALG = 1
EXP = 0
flag=0


if ALG == 1:
    for half in [1]:
    # for half in [0]:
        if half==0:
            outfile = './out-float'
        else:
            outfile = './out-half'
        fo = open(outfile, 'w')
        fo.close()
        norm_file = open('normALG', 'r')
        # for matrixdim in MADIM:
        for matrixdim in MADIM:
            # for i in range(0, 6):
            for i in range(0, 4):
                line = norm_file.readline()
                if line == '\n':
                    line = norm_file.readline()
                # for devicedim in [1,2,4,8]:
                for devicedim in [1]:
                    f = open('para.h', 'w')
                    f.write("#define USINGHALF "+str(half)+"\n")
                    f.write("const double NormINIT="+line+";"+"\n")
                    f.write("#define TESTTIME 4"+"\n")
                    f.write("#define WARMUP 2"+"\n")
                    f.write("const int inM="+str(matrixdim)+";"+"\n")
                    f.write("const int inK="+str(matrixdim)+";"+"\n")
                    f.write("const int inN="+str(matrixdim)+";"+"\n")
                    f.write("#define DEVICEDIM "+str(devicedim)+"\n")
                    f.write("#define CNN 0"+"\n")
                    f.write("#define DECAY 0"+"\n")
                    f.write("#define MATRIXNOR 0"+"\n")
                    f.write("#define MATRIXEXP 0"+"\n")
                    f.write("#define MATRIXALG 1"+"\n")

                    f.write("#if DECAY"+"\n")
                    f.write("const std::string FILENAMEA=\"a\";\n")
                    f.write("const std::string FILENAMEB=\"b\";\n")
                    f.write("#endif"+"\n")

                    f.write("#if CNN"+"\n")
                    f.write("const std::string FILENAMEA=\"data_cnn/conv_w_col.csv(64, 576).csv\";"+"\n")
                    f.write("const std::string FILENAMEB=\"data_cnn/conv_X_col.csv(576, 102400).csv\";"+"\n")
                    f.write("#endif"+"\n")
                    f.close()
                    os.system("./run.sh>>"+outfile)
        norm_file.close()
        os.system("echo end")

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