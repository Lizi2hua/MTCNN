import json
import glob
import os

#用于生成txt文件

def read_json(json_path):
    i=0
    dir=glob.glob(os.path.join(json_path,"*.json"))
    # print(dir)
    label_dict={}
    for lable_path in dir:

        label=open(lable_path,encoding='utf-8')
        label=json.load(label)

        object,label=label['path'],label["outputs"]["object"][0]['bndbox']
        print(object,label)
        # exit()
        print(i)
        i+=1
        object=os.path.split(object)[-1]
        label=[label["xmin"],label["ymin"],label["xmax"],label["ymax"]]
        label_dict1={object:label}
        label_dict.update(label_dict1)

    return label_dict

json_path=r"C:\Users\Administrator\Desktop\test1000\label"
b=read_json(json_path)

#将文件写为一个.txt文件
path=r"C:\Users\Administrator\Desktop\test1000"
label_txt=os.path.join(path,'label.txt')
label_txt=open(label_txt,'w')
for data in b:
    print(data)
    print(b[data])
    label_data=str(data)+' '+str(b[data][0])+' '+str(b[data][1])+' '+str(b[data][2])+' '+str(b[data][3])+'\n'
    print(label_data)
    label_txt.write(label_data)
label_txt.close()




