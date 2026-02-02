import pandas as pd
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def compute_angle(a, b, c):
    origin = np.zeros(2)
    ab = (b-origin) - (a-origin)
    ac = (c-origin) - (a-origin)
    if (np.linalg.norm(ab) * np.linalg.norm(ac)) > 0:
        theta = np.arccos(np.dot(ab, ac) / (np.linalg.norm(ab) * np.linalg.norm(ac)))
    else:
        theta = 0
    return theta, 2*np.pi - theta


plt.rcParams["font.family"] = "Times New Roman"
plt.rc('text', usetex=True)

plt.ion()
TJunction = pd.read_csv('boundary.csv',delimiter=',',header=None)
boundingBox1 = pd.read_csv('boundary_1.csv',delimiter=',',header=None)
boundingBox2 = pd.read_csv('boundary_2.csv',delimiter=',',header=None)
boundingBox3 = pd.read_csv('boundary_3.csv',delimiter=',',header=None)

fixedPoint1 = pd.read_csv('fixed_1.csv',delimiter=',',header=None)
fixedPoint2 = pd.read_csv('fixed_2.csv',delimiter=',',header=None)
fixedPoint3 = pd.read_csv('fixed_3.csv',delimiter=',',header=None)

fixedPoints1 =[]
fixedPoints2 =[]
fixedPoints3 =[]

for (i,j),val in np.ndenumerate(np.array(fixedPoint1)):
    fixedPoints1.append(val)
fixedPoints1array = np.array(boundingBox1)[fixedPoints1]

for (i,j),val in np.ndenumerate(np.array(fixedPoint2)):
    fixedPoints2.append(val)
fixedPoints2array = np.array(boundingBox2)[fixedPoints2]

for (i,j),val in np.ndenumerate(np.array(fixedPoint3)):
    fixedPoints3.append(val)
fixedPoints3array = np.array(boundingBox3)[fixedPoints3]

"""
for i in range(0,len(fixedPoint1)):
    fixedPoints1.append([boundingBox1.iloc[fixedPoint1.iloc[i,0],0],boundingBox1.iloc[fixedPoint1.iloc[i,0],1]])
    fixedPoints1array = np.asarray(fixedPoints1)

for i in range(0,len(fixedPoint2)):
    fixedPoints2.append([boundingBox2.iloc[fixedPoint2.iloc[i,0],0],boundingBox2.iloc[fixedPoint2.iloc[i,0],1]])
    fixedPoints2array = np.asarray(fixedPoints2)

for i in range(0,len(fixedPoint3)):
    fixedPoints3.append([boundingBox3.iloc[fixedPoint3.iloc[i,0],0],boundingBox3.iloc[fixedPoint3.iloc[i,0],1]])
    fixedPoints3array = np.asarray(fixedPoints3)
"""

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('x (m)', fontsize = 20,  fontname = 'Times New Roman',style='italic')
ax1.set_ylabel('y (m)', fontsize = 20, fontname = 'Times New Roman',style='italic')
for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(17.5)
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(17.5)

ax1.plot(TJunction.iloc[:,0],TJunction.iloc[:,1], label='T-Junction boundary', linewidth=2.5,c='k')
ax1.plot(boundingBox1.iloc[:,0],boundingBox1.iloc[:,1],'--',label='Bounding-box 1', marker='o', linewidth=2,c='b')
ax1.plot(boundingBox2.iloc[:,0],boundingBox2.iloc[:,1],'--', label='Bounding-box 2', marker='o', linewidth=2,c='r')
ax1.plot(boundingBox3.iloc[:,0],boundingBox3.iloc[:,1],'--', label='Bounding-box 3', marker='o', linewidth=2,c='m')

ax1.scatter(fixedPoints1array[:,0],fixedPoints1array[:,1],label='Fixed points', marker='s', s=80, c='k')
ax1.scatter(fixedPoints2array[:,0],fixedPoints2array[:,1], marker='s',s=80, c='k')
ax1.scatter(fixedPoints3array[:,0],fixedPoints3array[:,1], marker='s',s=80, c='k')
ax1.set(ylim=[-0.8, 0.8])
leg = plt.legend()
plt.setp(ax1.get_legend().get_texts(), fontsize='20')
plt.tight_layout()
