import os
import json

for eachf in os.listdir("./co"):
    print(eachf)
    f = open(os.path.join("./co",eachf), 'r')
    _ = eachf.find("_")
    if _==12:
        num = eachf[_-1]
    elif _==13:
        num = eachf[_-2:_]
    elif _==8:
        num = eachf[_-1]
    elif _==9:
        num = eachf[_-2:_]
    part = eachf[_+1]
    o = open('nus_coocurrence'+num+'_'+part+'.json', 'w')
    dic = json.load(open('category_nus.json', 'r'))
    outlist = []
    for eachline in f.readlines():
        eachidxs = []
        eachcluster = eachline.replace("[",'').replace("]",'').replace("\'",'').replace("\n",'').split(",")
        print(eachcluster) 
        for x in eachcluster:
            if x[0] == ' ': 
                x=x[1:]   
            eachidxs.append(str(dic[x]))  
        outlist.append(dict(zip(eachidxs,range(len(eachidxs)))))
    print(str(outlist).replace("\'","\""))
    o.write(str(outlist).replace("\'","\""))
    f.close()
    o.close()