f = open("in")
fo = open('out', 'w')

lines = f.readlines()
i = 0
for line in lines:

    list = line.split("=")
    if list[0] == "time":
        fo.write(list[1].split("s")[0]+"\n")
        i += 1
        if i % 8 == 0:
            fo.write("\n")

    # list = line.split("=")
    # if list[0] == "EF":
    #     fo.write(list[3].split(" ")[0])
    #     i += 1
    #     if i % 6 == 0:
    #         fo.write("\n")

    # list = line.split("=")
    # if list[0] == "time":
    #     fo.write(list[1].split("s")[0]+"\n")
    #     i += 1
    #     if i % 3 == 0:
    #         fo.write("\n")
    #     if i % (18) == 0:
    #         fo.write("---------------------------\n")

f.close()
