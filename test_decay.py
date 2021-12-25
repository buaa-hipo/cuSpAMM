import os
MADIM = [1024, 2048, 4096, 4096*2, 4096*4, 4096*8]
E = [1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10,1e-16,1e-32]
LoNum = 32
TESTTIME = 20
ALG = 0
EXP = 1
flag=0
matrixdim=13656

# for file in os.listdir("mtxdata"):
#     print(file)

if 1:
    for half in [0,1]:
    # for half in [0]:
        if half==0:
            outfile = 'out-half'
        else:
            outfile = 'out-float'
        fo = open(outfile, 'w')
        fo.close()
        for file in os.listdir("data_decay"):
            file = '\"' + "data_decay/"+file + '\"'
            matrixdim=file.split("_")[-1].split(".mtx")[0]
            # print(matrixdim)
            for i in range(0, 6):
            # for i in range(0, 1):
                for devicedim in [1,2,4,8]:
                # for devicedim in [1,2]:
                    f = open('para.h', 'w')
                    f.write("#define USINGHALF "+str(half)+"\n")
                    f.write("const double NormINIT="+str(E[i])+";"+"\n")
                    f.write("#define TESTTIME 20"+"\n")
                    f.write("#define WARMUP 3"+"\n")
                    f.write("const int inM="+matrixdim+";"+"\n")
                    f.write("const int inK="+matrixdim+";"+"\n")
                    f.write("const int inN="+matrixdim+";"+"\n")
                    f.write("#define DEVICEDIM "+str(devicedim)+"\n")
                    
                    f.write("#define CNN 0"+"\n")
                    f.write("#define DECAY 1"+"\n")
                    f.write("#define MATRIXNOR 0"+"\n")
                    f.write("#define MATRIXEXP 0"+"\n")
                    f.write("#define MATRIXALG 0"+"\n")

                    f.write("#if DECAY"+"\n")
                    f.write("const std::string FILENAMEA="+file+";\n")
                    f.write("const std::string FILENAMEB="+file+";\n")
                    f.write("#endif"+"\n")

                    f.write("#if CNN"+"\n")
                    f.write("const std::string FILENAMEA=\"data_cnn/conv_w_col.csv(64, 576).csv\";"+"\n")
                    f.write("const std::string FILENAMEB=\"data_cnn/conv_X_col.csv(576, 102400).csv\";"+"\n")
                    f.write("#endif"+"\n")

                    f.close()
                    os.system("./run.sh>>"+outfile)

