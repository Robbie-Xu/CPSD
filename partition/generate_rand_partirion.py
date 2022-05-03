import random
import os
for j in range(5):
    for subnum in [2,3,5,8,13]:
        idxs = list(range(81))
        random.shuffle(idxs)
        print(idxs)
        for i in range(len(idxs)):
            idxs[i] = str(idxs[i])
        cut_list = []
        for each in range(subnum-1):
            cut = random.randint(0,80)
            if cut not in cut_list:
                cut_list.append(cut)
            else:
                cut = random.randint(0,80)
                if cut not in cut_list:
                    cut_list.append(cut)
                else:
                    assert 1==2
        cut_list.append(81)
        assert len(cut_list) == subnum
        cut_list.sort()
        dic = []
        t=0
        for i in range(subnum):
            dic.append(dict(zip(idxs[t:cut_list[i]],range(len(idxs[t:cut_list[i]])))))
            t = cut_list[i]
        with open("nus_random"+str(subnum)+"_"+str(j)+".json",'w') as f:
            f.write(str(dic).replace("\'","\""))
    

    