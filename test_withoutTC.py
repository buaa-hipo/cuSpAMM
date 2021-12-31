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

normlist=[]
norm_file = open('normALG16', 'r')
lines = norm_file.readlines()
for line in lines:
    list=line.replace("\n","").split(",")
    normlist.append(list)

print(normlist)

if 1:
    outfile = 'out_withoutTC'
    fo = open(outfile, 'w')
    fo.close()
    #每个dim
    for dimno in range(0,6):
        matrixdim=MADIM[dimno]
        #每个error
        for eno in range(0, 4):
            norm=normlist[eno][dimno]
            devicedim=1
            f = open('para.h', 'w')
            f.write("#define USINGHALF "+str(1)+"\n")
            f.write("const double NormINIT="+norm+";"+"\n")
            f.write("#define TESTTIME 20"+"\n")
            f.write("#define WARMUP 3"+"\n")
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
            # f.write("const std::string FILENAMEA="+file+";\n")
            # f.write("const std::string FILENAMEB="+file+";\n")
            f.write("#endif"+"\n")

            f.write("#if CNN"+"\n")
            f.write("const std::string FILENAMEA=\"data_cnn/conv_w_col.csv(64, 576).csv\";"+"\n")
            f.write("const std::string FILENAMEB=\"data_cnn/conv_X_col.csv(576, 102400).csv\";"+"\n")
            f.write("#endif"+"\n")

            f.close()
            os.system("./run.sh>>"+outfile)

