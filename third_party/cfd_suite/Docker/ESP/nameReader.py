## Das script ist nur zur Vorbereitung nötig und sollte nicht
## auf dem finalen Stick für paris liegen

import sys

with open("masterSTL.stl","r") as myFile:
    lines = myFile.readlines()

names = []  
for line in lines:
    if "solid" in line and not "endsolid" in line:
        name = line.replace("solid","").replace("\n","").replace(" ","")
        removeList = ["wall","symPlane","wNE","a0","a1"]
        if name not in removeList:
            names.append(name)

names.sort()

for name in names:
    sys.stdout.write("\""+name + "\",")
    

print(len(names))
        
