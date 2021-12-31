f = open('./res_result/cf_cnn_fp16', 'r')
lines = f.readlines()
liness=""

for line in lines:
    liness+=line

lines = liness.split("INIT DONE------------")
for block in lines[1:]:
    list=block.split("\n")
    M=list[1].split("M=")[1].split(" ")[0]
    device=list[1].split("DEVICE=")[1].split(" ")[0]
    device=int(device)
    time=list[4].split("spammm time=")[1].split("s")[0]
    ratio=list[9].split("rate=")[1].split("% ")[0]
    if device == 1:
        print(M+" "+ratio+" "+time)
