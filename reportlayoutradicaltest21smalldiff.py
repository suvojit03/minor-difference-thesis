import json
import os
import dash
from dash_bootstrap_components._components.Col import Col
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
# import dash_table
import pandas as pd
import plotly.graph_objects as go
import plotly       #(version 4.4.1)
import plotly.express as px
import numpy as np
from math import sin, cos, radians, pi, atan2, degrees
from functools import reduce
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

import base64
from flask import Flask, Response
import dash_player
import math
#import video-engine as rpd
from textwrap import dedent
import plotly.io as pio
from dash import no_update
from sklearn.preprocessing import StandardScaler

import dash_player
import dash

# import dash_core_components as dcc
import numpy as np
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_daq as daq



pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
# pd.set_option('display.float_format', '{:.2f}'.format)



df = pd.read_excel('TEACHERV3_pos.xlsx')
#print (df)
df1 = pd.read_excel('TEACHERV3_rot.xlsx')
#print(df1)
# df2 = pd.read_excel('capoeira800framesmodifiedv3_rot.xlsx')
df2 = pd.read_excel('STUDENTV3MINORmodified_rotV2.xlsx')
df3 = pd.read_excel('STUDENTV3MINORmodified_pos_copyV2.xlsx')

#merginf both position and rotation tables based on key
# frames = [df, df1]
#df2 = pd.concat(frames,join = 'inner', axis = 1,keys='time')
df4 = pd.merge(df, df1, on=['time'])
df5 = pd.merge(df2, df3, on=['time'])
df7 = pd.merge(df4, df5, on=['time'])

df7["hipsdiff"]=""
df7["leftlegupdiff"]=""
df7["leftlegdiff"]=""
df7["leftfootdiff"]=""
df7["lefttoebasediff"]=""
df7["hipsdiff"]=""
df7["rightlegupdiff"]=""
df7["rightlegdiff"]=""
df7["Rightfootdiff"]=""
df7["Righttoebasediff"]=""
df7["Spinediff"]=""
df7["Spine1diff"]=""
df7["Spine2diff"]=""
df7["Spine3diff"]=""
df7["Spine4diff"]=""
df7["leftshoulderdiff"]=""
df7["leftarmdiff"]=""
df7["leftforearmdiff"]=""
df7["lefthanddiff"]=""
df7["diffstandard"]=""
df7["diffnormal"]=""
df7['leftlegupdiffnormal']=""
df7['rightlegupdiffnormal']=""
df7['leftlegdiffnormal']=""
df7['rightlegdiffnormal']=""
df7['leftshoulderdiffnormal']=""
df7['rightshoulderdiffnormal']=""
df7["leftforearmdiffnormal"]=""
df7["rightforearmdiffnormal"]=""
df7["leftarmdiffnormal"] =""
df7["rightarmdiffnormal"] =""
df7["diffuppernormalsee"] =""
df7["difflowernormalsee"] =""






df7.to_csv('dataframe.csv')


def rotmatrix1(a,b,c,d,e,f):    #CALCULATING FROBENIUS NORM OF HIPS between 2 users
    #print("rotation matrix  for user 1 _x")
    c1 = cos(float(a))
    c2 = cos(float(b))
    c3 = cos(float(c))
    s1 = sin(float(a))
    s2 = sin(float(b))
    s3 = sin(float(c))


           #creating the translation matrix for hips for user 1

#print(z1)


    r1 = float(d)
    r2 = float(e)
    r3 = float(f)

    rotx = [[1,0,0],[0,c1,-s1],[0,s1,c1]]
    roty = [[c2,0,s2],[0,1,0],[-s2,0,c2]]
    rotz = [[c3,-s3,0],[s3,c3,0],[0,0,1]]
    rotxyz= np.linalg.multi_dot([rotx,roty,rotz])
   
   
    rotxyz1 = rotxyz.reshape(3,3)
 
    column_translation = np.array([r1, r2, r3])
    translation = np.column_stack((rotxyz1, column_translation))
    row_to_be_added = np.array([0,0,0,1])

    rot = np.vstack((translation, row_to_be_added) )
    #np.set_printoptions(suppress=True)

    rotmat1 = np.array(rot,dtype=float)
    # rotmat1 = rotmat.reshape(4,4)
  
     
    return rotmat1 #rotmat1 is translation matrix for user1


def rotmatrix2(g,h,i,j,k,l):

    #print("rotation matrix for hips for user 2 _y")



    c11 = cos(float(g))
    c21 = cos(float(h))
    c31 = cos(float(i))
    s11 = sin(float(g))
    s21 = sin(float(h))
    s31 = sin(float(i))
    z11 = (c11*c31) - (s11*s21*s31)
    z21 = (c31*s11) + (c11*s21*s31)
    z31 =-(c21*s31)
    x11 = (c21*s11)
    x21 = (c11*c21)
    x31 =  s21
    y11 = (c11*s31) + (c31*s11*s21)
    y21 = (s11*s31) - (c11*c31*s21)
    y31 = (c21*c31)
    y41 = (c11*c31) - (s11*s21*s31)
    y51 = -(c21)*s31
    y61 = (s11*s31) - (c11*c31*s21)



    r11 = float(j)
    r21 = float(k)
    r31 = float(l)

    rot1 = [y31, z31, s21,r11,y11,y41,y51,r21,y61,z21,x21,r31,0,0,0,1]
    #np.set_printoptions(suppress=True)

    rotmat11 = np.array(rot1,dtype=float)
    rotmat2 = rotmat11.reshape(4,4) #rotmat2 is translation matrix of user2

    return rotmat2

def frob(x,y):  #calculating frobenius norm
    #print("calculating inverse of matrix of user 2")

    zinv = (np.linalg.inv(x))
    xinv = zinv.astype(np.float)
    y = y.astype(np.float)
   # print(rotmat2i)
    #print("Difference matrix is")
    D = (np.matmul(y,xinv))
    D = D.astype(np.float)
    #print(D)

   # print("calculating frobenius norm of Difference matrix")
    D1= float(np.linalg.norm(D, 'fro'))
    #print(D1)
    return (D1)
#print(df)

diff1 = 0
diff2 = 0
diff3 = 0
diff4 = 0
diff5 = 0

leftlegupdiff1 = 0
leftlegupdiff2 = 0
leftlegupdiff3 = 0
leftlegupdiff4 = 0
leftlegupdiff5 = 0

leftlegdiff1 = 0
leftlegdiff2 = 0
leftlegdiff3 = 0
leftlegdiff4 = 0
leftlegdiff5 = 0

rightlegupdiff1 = 0
rightlegupdiff2 = 0
rightlegupdiff3 = 0
rightlegupdiff4 = 0
rightlegupdiff5 = 0

rightlegdiff1 = 0
rightlegdiff2 = 0
rightlegdiff3 = 0
rightlegdiff4 = 0
rightlegdiff5 = 0

LeftShoulderdiff1 = 0
LeftShoulderdiff2 = 0
LeftShoulderdiff3 = 0
LeftShoulderdiff4 = 0
LeftShoulderdiff5 = 0

RightShoulderdiff1 = 0
RightShoulderdiff2 = 0
RightShoulderdiff3 = 0
RightShoulderdiff4 = 0
RightShoulderdiff5 = 0


LeftArmdiff1 = 0
LeftArmdiff2 = 0
LeftArmdiff3 = 0
LeftArmdiff4 = 0
LeftArmdiff5 = 0

LeftForeArmdiff1 = 0
LeftForeArmdiff2 = 0
LeftForeArmdiff3 = 0
LeftForeArmdiff4 = 0
LeftForeArmdiff5 = 0

RightArmdiff1 = 0
RightArmdiff2 = 0
RightArmdiff3 = 0
RightArmdiff4 = 0
RightArmdiff5 = 0

RightForeArmdiff1 = 0
RightForeArmdiff2 = 0
RightForeArmdiff3 = 0
RightForeArmdiff4 = 0
RightForeArmdiff5 = 0




for i,row in df7.iterrows():

    diff = 0
    diffleftleg = 0
    diffrightleg = 0
    difflefthand = 0
    diffrighthand = 0
    hipsfrob = 0
    leftlegfrob = 0
    leftlegupfrob = 0
    leftfootfrob = 0
    lefttoebasefrob = 0


    #print(row['Hipszz_x'])
    hipsrotmat1=[]
    hipsrotmat2=[]
    Spinerotmat1=[]
    Spinerotmat2=[]
    Spine1rotmat1=[]
    Spine1rotmat2=[]
    Spine2rotmat1=[]
    Spine2rotmat2=[]
    Spine3rotmat1=[]
    Spine3rotmat2=[]
    Spine4rotmat1=[]
    Spine4rotmat2=[]
    Headrotmat1=[]
    Headrotmat2=[]
    Neckrotmat1=[]
    Neckrotmat2=[]
    LeftShoulderrotmat1=[]
    LeftShoulderrotmat2=[]
    LeftArmrotmat1=[]
    LeftArmrotmat2=[]
    LeftForeArmrotmat1=[]
    LeftForeArmrotmat2=[]
    LeftHandrotmat1=[]
    LeftHandrotmat2=[]
    leftleguprotmat1 = []
    leftleguprotmat2 = []
    leftlegrotmat1 = []
    leftlegrotmat2 = []
    leftfootrotmat1 = []
    leftfootrotmat2 = []
    LeftToeBaserotmat1 = []
    LeftToeBaserotmat2 =[]


########################################## Hips ###################################
    diff = 0
    
    # row['Hipsz_x'] =  row['Hipsz_x'] - 16.6906
    


    hipsrotmat1 = rotmatrix1(row['Hipsxx_x'],row['Hipsyy_x'],row['Hipszz_x'],row['Hipsx_x'],row['Hipsy_x'],row['Hipsz_x'])

    hipsrotmat2 = rotmatrix1(row['Hipsxx_y'],row['Hipsyy_y'],row['Hipszz_y'],row['Hipsx_y'],row['Hipsy_y'],row['Hipsz_y'])
    hipsfrob = frob(hipsrotmat1,hipsrotmat2)
    # hipsfrob = hipsfrob - 2
    # if (hipsfrob < 0):
    #     hipsfrob = 0
    # hipsfrob = round(hipsfrob,2)
    df7.loc[i,'hipsdiff'] = hipsfrob
    

    diff = diff + hipsfrob
    diffleftleg = diffleftleg + hipsfrob
    diffrightleg = diffrightleg + hipsfrob
    difflefthand = difflefthand + hipsfrob
    diffrighthand = diffrighthand + hipsfrob
    hipsfrob = 0

#######################################leftleg####################################################
    #leftlegup’
    # row['LeftUpLegz_x'] = row['LeftUpLegz_x'] - 16.6906

    leftleguprotmat1 = rotmatrix1(row['LeftUpLegxx_x'],row['LeftUpLegyy_x'],row['LeftUpLegzz_x'],row['LeftUpLegx_x'],row['LeftUpLegy_x'],row['LeftUpLegz_x'])
    leftleguprotmat2 = rotmatrix1(row['LeftUpLegxx_y'],row['LeftUpLegyy_y'],row['LeftUpLegzz_y'],row['LeftUpLegx_y'],row['LeftUpLegy_y'],row['LeftUpLegz_y'])
    leftleguprotmat1 = np.matmul(hipsrotmat1,leftleguprotmat1)
    leftleguprotmat2 = np.matmul(hipsrotmat2,leftleguprotmat2)
    leftlegupfrob = frob(leftleguprotmat1,leftleguprotmat2)
    # leftlegupfrob = leftlegupfrob - 2
    
    # if (leftlegupfrob < 0):
    #     leftlegupfrob = 0

    leftlegupdiff = leftlegupfrob

    diff = diff + leftlegupfrob
    diffleftleg = diffleftleg + leftlegupfrob
    leftlegupfrob = 0

    # #leftleg

    # row['LeftLegz_x'] = row['LeftLegz_x'] - 16.6906

    leftlegrotmat1 = rotmatrix1(row['LeftLegxx_x'],row['LeftLegyy_x'],row['LeftLegzz_x'],row['LeftLegx_x'],row['LeftLegy_x'],row['LeftLegz_x'])
    leftlegrotmat2 = rotmatrix1(row['LeftLegxx_y'],row['LeftLegyy_y'],row['LeftLegzz_y'],row['LeftLegx_y'],row['LeftLegy_y'],row['LeftLegz_y'])
    leftlegrotmat1 = np.linalg.multi_dot([hipsrotmat1,leftleguprotmat1,leftlegrotmat1])
    leftlegrotmat2 = np.linalg.multi_dot([hipsrotmat2,leftleguprotmat2,leftlegrotmat2])
    leftlegfrob = frob(leftlegrotmat1,leftlegrotmat2)
    # leftlegfrob = leftlegfrob - 2

    # if (leftlegfrob < 0):
    #     leftlegfrob = 0

    
    leftlegdiff = leftlegfrob


    diff = diff + leftlegfrob
    diffleftleg = diffleftleg + leftlegfrob
    leftlegfrob = 0

   
 ##########################################Right Leg###################################   

    #Rightlegup’
    # row['RightUpLegz_x'] = row['RightUpLegz_x'] - 16.6906
    Rightleguprotmat1 = rotmatrix1(row['RightUpLegxx_x'],row['RightUpLegyy_x'],row['RightUpLegzz_x'],row['RightUpLegx_x'],row['RightUpLegy_x'],row['RightUpLegz_x'])
    Rightleguprotmat2 = rotmatrix1(row['RightUpLegxx_y'],row['RightUpLegyy_y'],row['RightUpLegzz_y'],row['RightUpLegx_y'],row['RightUpLegy_y'],row['RightUpLegz_y'])
    Rightleguprotmat1 = np.matmul(hipsrotmat1,Rightleguprotmat1)
    Rightleguprotmat2 = np.matmul(hipsrotmat2,Rightleguprotmat2)
    Rightlegupfrob = frob(Rightleguprotmat1,Rightleguprotmat2)
    # Rightlegupfrob = Rightlegupfrob - 2

    # if (Rightlegupfrob<0):
    #     Rightlegupfrob = 0
    rightlegupdiff = Rightlegupfrob


    diff = diff + Rightlegupfrob
    diffrightleg = diffrightleg + Rightlegupfrob

    df7['rightlegupdiff']= pd.to_numeric(df7['rightlegupdiff'])

    # #Rightleg
    # row['RightLegz_x'] = row['RightLegz_x'] - 16.6906

    Rightlegrotmat1 = rotmatrix1(row['RightLegxx_x'],row['RightLegyy_x'],row['RightLegzz_x'],row['RightLegx_x'],row['RightLegy_x'],row['RightLegz_x'])
    Rightlegrotmat2 = rotmatrix1(row['RightLegxx_y'],row['RightLegyy_y'],row['RightLegzz_y'],row['RightLegx_y'],row['RightLegy_y'],row['RightLegz_y'])
    Rightlegrotmat1 = np.linalg.multi_dot([hipsrotmat1,Rightleguprotmat1,Rightlegrotmat1])
    Rightlegrotmat2 = np.linalg.multi_dot([hipsrotmat2,Rightleguprotmat2,Rightlegrotmat2])
    Rightlegfrob = frob(Rightlegrotmat1,Rightlegrotmat2)
    # Rightlegfrob = Rightlegfrob - 2

    # # Rightlegfrob = format(Rightlegfrob, '.0f')
    

    # if (Rightlegfrob<0):
    #     Rightlegfrob = 0


    
    rightlegdiff = Rightlegfrob

    # df7['rightlegdiff']= pd.to_numeric(df7['rightlegdiff'])

    diff = diff + Rightlegfrob
    diffrightleg = diffrightleg + Rightlegfrob

            

    ######################### Spine #################################################

    # row['Spinez_x'] = row['Spinez_x'] - 16.6906

    Spinerotmat1 = rotmatrix1(row['Spinexx_x'],row['Spineyy_x'],row['Spinezz_x'],row['Spinex_x'],row['Spiney_x'],row['Spinez_x'])
    Spinerotmat2 = rotmatrix1(row['Spinexx_y'],row['Spineyy_y'],row['Spinezz_y'],row['Spinex_y'],row['Spiney_y'],row['Spinez_y'])
    Spinerotmat1 = np.linalg.multi_dot([hipsrotmat1,Spinerotmat1])
    Spinerotmat2 = np.linalg.multi_dot([hipsrotmat2,Spinerotmat2])
    Spinefrob = frob(Spinerotmat1,Spinerotmat2)
    # Spinefrob = Spinefrob - 2


    # if (Spinefrob < 0):
    #     Spinefrob = 0

    
    df7.loc[i,'Spinediff']= Spinefrob


    diff = diff + Spinefrob
    difflefthand = difflefthand + Spinefrob
    diffrighthand = diffrighthand + Spinefrob
    Spinefrob = 0

    ######################### Spine1 #################################################
    # row['Spine1z_x'] = row['Spine1z_x'] - 16.6906
    Spine1rotmat1 = rotmatrix1(row['Spine1xx_x'],row['Spine1yy_x'],row['Spine1zz_x'],row['Spine1x_x'],row['Spine1y_x'],row['Spine1z_x'])
    Spine1rotmat2 = rotmatrix1(row['Spine1xx_y'],row['Spine1yy_y'],row['Spine1zz_y'],row['Spine1x_y'],row['Spine1y_y'],row['Spine1z_y'])
    Spine1rotmat1 = np.linalg.multi_dot([hipsrotmat1,Spinerotmat1,Spine1rotmat1])
    Spine1rotmat2 = np.linalg.multi_dot([hipsrotmat2,Spinerotmat2, Spine1rotmat2])
    Spine1frob = frob(Spine1rotmat1,Spine1rotmat2)
    # Spine1frob = Spine1frob - 2

    # if (Spine1frob < 0):
    #     Spine1frob = 0

    
    df7.loc[i,'Spine1diff']= Spine1frob

    diff = diff + Spine1frob
    difflefthand = difflefthand + Spine1frob
    diffrighthand = diffrighthand + Spine1frob
    Spine1frob = 0


######################### Spine2 #################################################
    # row['Spine2z_x'] = row['Spine2z_x'] - 16.6906
    
    Spine2rotmat1 = rotmatrix1(row['Spine2xx_x'],row['Spine2yy_x'],row['Spine2zz_x'],row['Spine2x_x'],row['Spine2y_x'],row['Spine2z_x'])
    Spine2rotmat2 = rotmatrix1(row['Spine2xx_y'],row['Spine2yy_y'],row['Spine2zz_y'],row['Spine2x_y'],row['Spine2y_y'],row['Spine2z_y'])
    Spine2rotmat1 = np.linalg.multi_dot([hipsrotmat1,Spinerotmat1,Spine1rotmat1,Spine2rotmat1])
    Spine2rotmat2 = np.linalg.multi_dot([hipsrotmat2,Spinerotmat2,Spine1rotmat2, Spine2rotmat2])
    Spine2frob = frob(Spine2rotmat1,Spine2rotmat2)
    # Spine2frob = Spine2frob - 2

    # if (Spine2frob < 0):
    #     Spine2frob = 0

    
    df7.loc[i,'Spine2diff']= Spine2frob

    diff = diff + Spine2frob
    difflefthand = difflefthand + Spine2frob
    diffrighthand = diffrighthand + Spine2frob
    Spine2frob = 0

######################### LeftShoulder #################################################

    # row['LeftShoulderz_x'] = row['LeftShoulderz_x'] - 16.6906
    
    LeftShoulderrotmat1 = rotmatrix1(row['LeftShoulderxx_x'],row['LeftShoulderyy_x'],row['LeftShoulderzz_x'],row['LeftShoulderx_x'],row['LeftShouldery_x'],row['LeftShoulderz_x'])
    LeftShoulderrotmat2 = rotmatrix1(row['LeftShoulderxx_y'],row['LeftShoulderyy_y'],row['LeftShoulderzz_y'],row['LeftShoulderx_y'],row['LeftShouldery_y'],row['LeftShoulderz_y'])
    LeftShoulderrotmat1 = np.linalg.multi_dot([hipsrotmat1,Spinerotmat1,Spine1rotmat1,Spine2rotmat1,LeftShoulderrotmat1])
    LeftShoulderrotmat2 = np.linalg.multi_dot([hipsrotmat2,Spinerotmat2,Spine1rotmat2,Spine2rotmat2, LeftShoulderrotmat2])
    LeftShoulderfrob = frob(LeftShoulderrotmat1,LeftShoulderrotmat2)
    # LeftShoulderfrob = LeftShoulderfrob - 2

    if (LeftShoulderfrob >3000):
        LeftShoulderfrob = 3000

    
    LeftShoulderdiff= LeftShoulderfrob

    diff = diff + LeftShoulderfrob
    difflefthand = difflefthand + LeftShoulderfrob
   
    LeftShoulderfrob = 0


######################### LeftArm #################################################
# spine1 and spine2 to remember####

    # row['LeftArmz_x'] =  row['LeftArmz_x'] - 16.6906
    
    LeftArmrotmat1 = rotmatrix1(row['LeftArmxx_x'],row['LeftArmyy_x'],row['LeftArmzz_x'],row['LeftArmx_x'],row['LeftArmy_x'],row['LeftArmz_x'])
    LeftArmrotmat2 = rotmatrix1(row['LeftArmxx_y'],row['LeftArmyy_y'],row['LeftArmzz_y'],row['LeftArmx_y'],row['LeftArmy_y'],row['LeftArmz_y'])
    LeftArmrotmat1 = np.linalg.multi_dot([hipsrotmat1,Spinerotmat1,Spine1rotmat1,Spine2rotmat1,LeftShoulderrotmat1,LeftArmrotmat1])
    LeftArmrotmat2 = np.linalg.multi_dot([hipsrotmat2,Spinerotmat2,Spine1rotmat2,Spine2rotmat2,LeftShoulderrotmat2,LeftArmrotmat2])
    LeftArmfrob = frob(LeftArmrotmat1,LeftArmrotmat2)
    # LeftArmfrob = LeftArmfrob - 2

    if (LeftArmfrob > 3000):
        LeftArmfrob = 3000

    
    LeftArmdiff = LeftArmfrob

    diff = diff + LeftArmfrob
    difflefthand = difflefthand + LeftArmfrob
    
    LeftArmfrob = 0

######################### LeftForeArm #################################################
    
    # row['LeftForeArmz_x'] = row['LeftForeArmz_x'] - 16.6906

    LeftForeArmrotmat1 = rotmatrix1(row['LeftForeArmxx_x'],row['LeftForeArmyy_x'],row['LeftForeArmzz_x'],row['LeftForeArmx_x'],row['LeftForeArmy_x'],row['LeftForeArmz_x'])
    LeftForeArmrotmat2 = rotmatrix1(row['LeftForeArmxx_y'],row['LeftForeArmyy_y'],row['LeftForeArmzz_y'],row['LeftForeArmx_y'],row['LeftForeArmy_y'],row['LeftForeArmz_y'])
    LeftForeArmrotmat1 = np.linalg.multi_dot([hipsrotmat1,Spinerotmat1,Spine1rotmat1,Spine2rotmat1,LeftShoulderrotmat1,LeftArmrotmat1,LeftForeArmrotmat1])
    LeftForeArmrotmat2 = np.linalg.multi_dot([hipsrotmat2,Spinerotmat2,Spine1rotmat2,Spine2rotmat2,LeftShoulderrotmat2,LeftArmrotmat2,LeftForeArmrotmat2])
    LeftForeArmfrob = frob(LeftForeArmrotmat1,LeftForeArmrotmat2)
    # LeftForeArmfrob = LeftForeArmfrob - 2

    if (LeftForeArmfrob > 3000):
        LeftForeArmfrob = 3000

    
    LeftForeArmdiff= LeftForeArmfrob

    diff = diff + LeftForeArmfrob
    difflefthand = difflefthand + LeftForeArmfrob
    LeftForeArmfrob = 0


######################### RightShoulder #################################################
    
    # row['RightShoulderz_x'] = row['RightShoulderz_x'] - 16.6906
    RightShoulderrotmat1 = rotmatrix1(row['RightShoulderxx_x'],row['RightShoulderyy_x'],row['RightShoulderzz_x'],row['RightShoulderx_x'],row['RightShouldery_x'],row['RightShoulderz_x'])
    RightShoulderrotmat2 = rotmatrix1(row['RightShoulderxx_y'],row['RightShoulderyy_y'],row['RightShoulderzz_y'],row['RightShoulderx_y'],row['RightShouldery_y'],row['RightShoulderz_y'])
    RightShoulderrotmat1 = np.linalg.multi_dot([hipsrotmat1,Spinerotmat1,Spine1rotmat1,Spine2rotmat1,RightShoulderrotmat1])
    RightShoulderrotmat2 = np.linalg.multi_dot([hipsrotmat2,Spinerotmat2,Spine1rotmat2,Spine2rotmat2, RightShoulderrotmat2])
    RightShoulderfrob = frob(RightShoulderrotmat1,RightShoulderrotmat2)
    # RightShoulderfrob = RightShoulderfrob - 2

    if (RightShoulderfrob > 3000):
        RightShoulderfrob = 3000

    
    RightShoulderdiff= RightShoulderfrob

    diff = diff + RightShoulderfrob
    diffrighthand = diffrighthand + RightShoulderfrob
    
    RightShoulderfrob = 0

######################### RightArm #################################################
####spine1 spine2 to remember########
    
    # row['RightArmz_x']=row['RightArmz_x'] - 16.6906
    RightArmrotmat1 = rotmatrix1(row['RightArmxx_x'],row['RightArmyy_x'],row['RightArmzz_x'],row['RightArmx_x'],row['RightArmy_x'],row['RightArmz_x'])
    RightArmrotmat2 = rotmatrix1(row['RightArmxx_y'],row['RightArmyy_y'],row['RightArmzz_y'],row['RightArmx_y'],row['RightArmy_y'],row['RightArmz_y'])
    RightArmrotmat1 = np.linalg.multi_dot([hipsrotmat1,Spinerotmat1,RightShoulderrotmat1,RightArmrotmat1])
    RightArmrotmat2 = np.linalg.multi_dot([hipsrotmat2,Spinerotmat2,RightShoulderrotmat2,RightArmrotmat2])
    RightArmfrob = frob(RightArmrotmat1,RightArmrotmat2)
    # RightArmfrob = RightArmfrob - 2

    if (RightArmfrob >3000):
        RightArmfrob = 3000

    
    RightArmdiff= RightArmfrob

    diff = diff + RightArmfrob
    
    diffrighthand = diffrighthand + RightArmfrob
    RightArmfrob = 0

######################### RightForeArm #################################################

    
    # row['RightForeArmz_x'] = row['RightForeArmz_x'] - 16.6906
    RightForeArmrotmat1 = rotmatrix1(row['RightForeArmxx_x'],row['RightForeArmyy_x'],row['RightForeArmzz_x'],row['RightForeArmx_x'],row['RightForeArmy_x'],row['RightForeArmz_x'])
    RightForeArmrotmat2 = rotmatrix1(row['RightForeArmxx_y'],row['RightForeArmyy_y'],row['RightForeArmzz_y'],row['RightForeArmx_y'],row['RightForeArmy_y'],row['RightForeArmz_y'])
    RightForeArmrotmat1 = np.linalg.multi_dot([hipsrotmat1,Spinerotmat1,Spine1rotmat1,Spine2rotmat1,RightShoulderrotmat1,RightArmrotmat1,RightForeArmrotmat1])
    RightForeArmrotmat2 = np.linalg.multi_dot([hipsrotmat2,Spinerotmat2,Spine1rotmat2,Spine2rotmat2,RightShoulderrotmat2,RightArmrotmat2,RightForeArmrotmat2])
    RightForeArmfrob = frob(RightForeArmrotmat1,RightForeArmrotmat2)
    # RightForeArmfrob = RightForeArmfrob - 2

    if (RightForeArmfrob > 3000):
        RightForeArmfrob = 3000

    
    RightForeArmdiff= RightForeArmfrob

    diff = diff + RightForeArmfrob
   
    diffrighthand = diffrighthand + RightForeArmfrob
    RightForeArmfrob = 0



    #########################################################################################

    difftotal = 0

    difftotal = (difflefthand + diffrighthand + diffleftleg + diffrightleg)/2

    diff = diff/14

    
    # if i % 3 == 1:
    #     diff1 = diff

    # if i % 3 == 2:
    #     diff2 = diff  

    # if i % 3 == 0:
    #     diff3 = diff 

    # diff6 = diff1 + diff2 + diff3 
    # diff = diff6/3

    # if i % 5 == 1:
    #     diff1 = diff
    #     leftlegupdiff1 = leftlegupdiff
    #     leftlegdiff1 = leftlegdiff
    #     rightlegupdiff1 = rightlegupdiff
    #     rightlegdiff1 = rightlegdiff
    #     LeftShoulderdiff1 = LeftShoulderdiff
    #     RightShoulderdiff1 = RightShoulderdiff
    #     LeftArmdiff1 = LeftArmdiff
    #     LeftForeArmdiff1 = LeftForeArmdiff
    #     RightArmdiff1 = RightArmdiff
    #     RightForeArmdiff1 = RightForeArmdiff



    # if i % 5 == 2:
    #     leftlegupdiff2 = leftlegupdiff
    #     leftlegdiff2 = leftlegdiff
    #     rightlegupdiff2 = rightlegupdiff
    #     rightlegdiff2 = rightlegdiff
    #     LeftShoulderdiff2 = LeftShoulderdiff
    #     RightShoulderdiff2 = RightShoulderdiff
    #     LeftArmdiff2 = LeftArmdiff
    #     LeftForeArmdiff2 = LeftForeArmdiff
    #     RightArmdiff2 = RightArmdiff
    #     RightForeArmdiff2 = RightForeArmdiff


    

    # if i % 5 == 3:
    #     diff3 = diff 
    #     leftlegupdiff3 = leftlegupdiff
    #     leftlegdiff3 = leftlegdiff
    #     rightlegupdiff3 = rightlegupdiff
    #     rightlegdiff3 = rightlegdiff
    #     LeftShoulderdiff3 = LeftShoulderdiff
    #     RightShoulderdiff3 = RightShoulderdiff
    #     LeftArmdiff3 = LeftArmdiff
    #     LeftForeArmdiff3 = LeftForeArmdiff
    #     RightArmdiff3 = RightArmdiff
    #     RightForeArmdiff3 = RightForeArmdiff





    # if i % 5 == 4:
    #     diff4 = diff  
    #     leftlegupdiff4 = leftlegupdiff
    #     leftlegdiff4 = leftlegdiff
    #     rightlegupdiff4 = rightlegupdiff
    #     rightlegdiff4 = rightlegdiff
    #     LeftShoulderdiff4 = LeftShoulderdiff
    #     RightShoulderdiff4 = RightShoulderdiff
    #     LeftArmdiff4 = LeftArmdiff
    #     LeftForeArmdiff4 = LeftForeArmdiff
    #     RightArmdiff4 = RightArmdiff
    #     RightForeArmdiff4 = RightForeArmdiff


  

    # if i % 5 == 0:
    #     diff5 = diff    
    #     leftlegupdiff5 = leftlegupdiff
    #     leftlegdiff5 = leftlegdiff
    #     rightlegupdiff5 = rightlegupdiff
    #     rightlegdiff5 = rightlegdiff
    #     LeftShoulderdiff5 = LeftShoulderdiff
    #     RightShoulderdiff5 = RightShoulderdiff
    #     LeftArmdiff5 = LeftArmdiff
    #     LeftForeArmdiff5 = LeftForeArmdiff
    #     RightArmdiff5 = RightArmdiff
    #     RightForeArmdiff5 = RightForeArmdiff





    # diff6 = diff1 + diff2 + diff3 + diff4 + diff5
    # diff = diff6/5 

    # leftlegupdiff6 = leftlegupdiff1 + leftlegupdiff2 + leftlegupdiff3 + leftlegupdiff4 + leftlegupdiff5
    # leftlegupdiff = leftlegupdiff6/5
    # if leftlegupdiff > 1500:
    #     leftlegupdiff = 1500

    # leftlegdiff6 = leftlegdiff1 + leftlegdiff2 + leftlegdiff3 + leftlegdiff4 + leftlegdiff5
    # leftlegdiff = leftlegdiff6/5
    # if leftlegdiff > 1500:
    #     leftlegdiff = 1500

    # rightlegupdiff6 = rightlegupdiff1 + rightlegupdiff2 + rightlegupdiff3 + rightlegupdiff4 + rightlegupdiff5
    # rightlegupdiff = rightlegupdiff6/5
    # if rightlegupdiff > 1500:
    #     rightlegupdiff = 1500

    # rightlegdiff6 = rightlegdiff1 + rightlegdiff2 + rightlegdiff3 + rightlegdiff4 + rightlegdiff5
    # rightlegdiff = rightlegdiff6/5
    # if rightlegdiff > 1500:
    #     rightlegdiff = 1500

    # LeftShoulderdiff6 = LeftShoulderdiff1 + LeftShoulderdiff2 + LeftShoulderdiff3 + LeftShoulderdiff4 + LeftShoulderdiff5
    # leftshoulderdiff = LeftShoulderdiff6/5
    # if leftshoulderdiff > 1500:
    #     leftshoulderdiff = 1500

    # RightShoulderdiff6 = RightShoulderdiff1 + RightShoulderdiff2 + RightShoulderdiff3 + RightShoulderdiff4 + RightShoulderdiff5
    # rightshoulderdiff = RightShoulderdiff6/5
    # if rightshoulderdiff > 1500:
    #     rightshoulderdiff = 1500



    # LeftArmdiff6 = LeftArmdiff1 + LeftArmdiff2 + LeftArmdiff3 + LeftArmdiff4 + LeftArmdiff5
    # leftarmdiff = LeftArmdiff6/5
    # if leftarmdiff > 1500:
    #     leftarmdiff = 1500
    

    # LeftForeArmdiff6 = LeftForeArmdiff1 + LeftForeArmdiff2 + LeftForeArmdiff3 + LeftForeArmdiff4 + LeftForeArmdiff5
    # leftforearmdiff = LeftForeArmdiff6/5
    # if leftforearmdiff > 1500:
    #     leftforearmdiff = 1500
    
  



    # RightArmdiff6 = RightArmdiff1 + RightArmdiff2 + RightArmdiff3 + RightArmdiff4 + RightArmdiff5
    # rightarmdiff = RightArmdiff6/5
    # if rightarmdiff > 1500:
    #     rightarmdiff = 1500
    
    

    # RightForeArmdiff6 = RightForeArmdiff1 + RightForeArmdiff2 + RightForeArmdiff3 + RightForeArmdiff4 + RightForeArmdiff5
    # rightforearmdiff = RightForeArmdiff6/5
    # if leftforearmdiff > 1500:
    #     leftforearmdiff = 1500
    
    


    
    if diff <=50:
        diffnormal=10
    if diff>50 and diff<=100:
        diffnormal=13
    if diff>100 and diff<=150:
        diffnormal=16
    if diff>150 and diff<=200:
        diffnormal=19
    if diff>200 and diff<=250:
        diffnormal=22
    if diff>250 and diff<=300:
        diffnormal=25
    if diff>300 and diff<=350:
        diffnormal=28
    if diff>350 and diff<=400:
        diffnormal=31
    if diff>400 and diff<=450:
        diffnormal= 34
    if diff>450 and diff<=500:
        diffnormal=37
    if diff>500 and diff<=550:
        diffnormal=40
    if diff>550 and diff<=600:
        diffnormal=43  
    if diff>600 and diff<=650:
        diffnormal=46
    if diff>650 and diff<=700:
        diffnormal=49
    if diff>700 and diff<=750:
        diffnormal=52
    if diff>750 and diff<=800:
        diffnormal=55
    if diff>800 and diff<=850:
        diffnormal=58
    if diff>850 and diff<=900:
        diffnormal=61
    if diff>900 and diff<=950:
        diffnormal=64
    if diff>950 and diff<=1000:
        diffnormal=67
    if diff>1000 and diff<=1050:
        diffnormal=70
    if diff>1050 and diff<=1100:
        diffnormal=73
    if diff>1100 and diff<=1150:
        diffnormal=76
    if diff>1150 and diff<=1200:
        diffnormal=79
    if diff>1200 and diff<=1250:
        diffnormal=82
    if diff>1250 and diff<=1300:
        diffnormal=85
    if diff>1300 and diff<=1350:
        diffnormal=88
    if diff>1350 and diff<=1400:
        diffnormal=91
    if diff>1400 and diff<=1450:
        diffnormal=94          
    if diff>1450 and diff<=1500:
        diffnormal=97  
    if diff>1500:
        diffnormal=100        



    df7.loc[i,'diff'] = diff
    df7.loc[i,'diffnormal'] = diffnormal
 
    df7.loc[i,'leftlegupdiff'] = leftlegupdiff

    df7.loc[i,'leftlegdiff'] = leftlegdiff

    df7.loc[i,'rightlegupdiff'] = rightlegupdiff
       
    df7.loc[i,'rightlegdiff'] = rightlegdiff
 
    df7.loc[i,'leftshoulderdiff'] = LeftShoulderdiff

    df7.loc[i,'rightshoulderdiff'] = RightShoulderdiff

    df7.loc[i,'leftarmdiff'] = LeftArmdiff
   
    df7.loc[i,'leftforearmdiff'] = LeftForeArmdiff
   
    df7.loc[i,'rightarmdiff'] = RightArmdiff

    df7.loc[i,'rightforearmdiff'] = RightForeArmdiff

   




    df7.loc[i,'diffinitial'] = diff
    # df7.loc[i,'difftotal'] = difftotal
    df7.loc[i,'difftotal'] = diff
    
    #print(df7)

    diffupper = difflefthand + diffrighthand
    difflower = diffleftleg + diffrightleg

    

    df7.loc[i,'diffupper'] = diffupper
    df7.loc[i,'difflower'] = difflower


#############  NORMALIZING TO 10 LEVEL ##############################################################################

start = 1
end = 100
width = end - start



# df7["diffnormal"] = ((df7["diff"] - df7["diff"].min())/(df7["diff"].max() - df7["diff"].min())) * width + start





df7["hipsdiffnormal"] = 0
df7["leftlegupdiffnormal"] = (df7["leftlegupdiff"] - df7["leftlegupdiff"].min())/(df7["leftlegupdiff"].max() - df7["leftlegupdiff"].min()) * width + start


df7["leftlegdiffnormal"] = (df7["leftlegdiff"] - df7["leftlegdiff"].min())/(df7["leftlegdiff"].max() - df7["leftlegdiff"].min()) * width + start

df7["leftarmdiffnormal"] = (df7["leftarmdiff"] - df7["leftarmdiff"].min())/(df7["leftarmdiff"].max() - df7["leftarmdiff"].min()) * width + start

df7["leftforearmdiffnormal"] = (df7["leftforearmdiff"] - df7["leftforearmdiff"].min())/(df7["leftforearmdiff"].max() - df7["leftforearmdiff"].min()) * width + start

df7["rightlegdiffnormal"] = (df7["rightlegdiff"] - df7["rightlegdiff"].min())/(df7["rightlegdiff"].max() - df7["rightlegdiff"].min()) * width + start

df7["rightlegupdiffnormal"] = (df7["rightlegupdiff"] - df7["rightlegupdiff"].min())/(df7["rightlegupdiff"].max() - df7["rightlegupdiff"].min()) * width + start


df7["rightarmdiffnormal"] = (df7["rightarmdiff"] - df7["rightarmdiff"].min())/(df7["rightarmdiff"].max() - df7["rightarmdiff"].min()) * width + start

df7["rightforearmdiffnormal"] = (df7["rightforearmdiff"] - df7["rightforearmdiff"].min())/(df7["rightforearmdiff"].max() - df7["rightforearmdiff"].min()) * width + start

# df7["diffuppernormal"] = ((df7["diffupper"] - df7["diffupper"].min())/(df7["diffupper"].max() - df7["diffupper"].min())) * width + start

# df7["difflowernormal"] = ((df7["difflower"] - df7["difflower"].min())/(df7["difflower"].max() - df7["difflower"].min())) * width + start






df7["leftshoulderdiffnormal"] = (df7["leftshoulderdiff"] - df7["leftshoulderdiff"].min())/(df7["leftshoulderdiff"].max() - df7["leftshoulderdiff"].min()) * width + start

df7["rightshoulderdiffnormal"] = (df7["rightshoulderdiff"] - df7["rightshoulderdiff"].min())/(df7["rightshoulderdiff"].max() - df7["rightshoulderdiff"].min()) * width + start

df7["diffuppernormalsee"] = (df7["leftshoulderdiffnormal"] + df7["rightshoulderdiffnormal"] + df7["rightforearmdiffnormal"]
+ df7["leftforearmdiffnormal"])/4

df7["difflowernormalsee"] = (df7["leftlegupdiffnormal"] + df7["leftlegdiffnormal"] + df7["rightlegupdiffnormal"]
+ df7["rightlegdiffnormal"])/4

external_stylesheets = ['https://codepen.io/anon/pen/mardKv.css']

app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])


theme =  {
    'dark': True,
    'detail': '#00EA64',
    'primary': '#00EA64',
    'secondary': '#00EA64',
}


#####################   NORMALIZING THE VALUES  ###########################################




   


upperbodyscore = 0
# ((df7['leftarmdiffnormal'].sum()) + (df7['leftforearmdiffnormal'].sum()) 
# + (df7['rightarmdiffnormal'].sum()) + (df7['rightforearmdiffnormal'].sum()) + 
# (df7['rightshoulderdiffnormal'].sum()) + (df7['leftshoulderdiffnormal'].sum()))/240

# lowerbodyscore = ((df7['leftlegupdiffnormal'].sum()) + (df7['leftlegdiffnormal'].sum()) + (df7['rightlegupdiffnormal'].sum()))/160

lowerbodyscore = 0






df8 = df7[["time","diff","diffnormal","leftlegdiffnormal","rightlegdiffnormal"]]
# df8 = df8.apply(pd.to_numeric)
# df8 = df8.apply(pd.to_numeric)


#error calculation condition


########### error is df9 ###############################################




total = df7['diff'].sum()
# total = total/400
# score = 10 - total
# print(score)
# print(total)
# lowerbodyscore = 0
# upperbodyscore = 0

dferror = df7[df7['diffnormal']>52]
z1 = len(dferror.index)
if z1 <= 40:
    score = 100

else:
       
    z = z1 - 40
    print(f'The scoreindex is {z1}.')
    score1 = 100 - ((z/400)*100)
    score = math.ceil(score1)
if score > 100:
    score = 100

dfuppererror = df7[df7['diffuppernormalsee']>52]
y1 = len(dfuppererror.index)
y = y1 - 0

print(f'The upperscoreindex  is {y}.')
upperbodyscore = 100 - ((y/400)*100)
# upperbodyscore = math.ceil(score1)


dflowererror = df7[df7['difflowernormalsee']>35]
x1 = len(dflowererror.index)
x = x1 - 20
print(f'The lowerscoreindex  is {x}.')
lowerbodyscore = 100 - ((x/400)*100)

if score == 100:
    lowerbodyscore = 100
    upperbodyscore = 100







df7.to_csv('hipsfinal.csv')
# df8.to_csv('difference.csv')
        


all_options = {
    'difference': ['difference'],
    'Upper Body': ['leftarmdiff','leftshoulderdiff','rightarmdiff','rightshoulderdiff'],
    'Lower Body': ['leftlegdiff','rightlegdiff']
    
    # 'clickonpart': {'clickonpart'}
}



# colors = {
#     'background': '#',
#     'text': '#00EA64'
# }

#00EA64

styles = {'pre': {'border': 'thin lightgrey solid', 'overflowX': 'scroll'}}

app.layout = html.Div(

    style={'color': 'white'},
    children = [
    html.Div(

    style={'height': '320px',
           'width':'1400px'},
        children = [
        
        dbc.Row
        (
            
            

            [

####################Score Start ####################################################                
                
                dbc.Col(
                                        
                    html.Div(
                    style =
                            {
                                'height' : '330px',
                                'width': '220px',
                                # 'background-color': '#000000',
                                'margin-left':'7px'
                            
                            
                            },    
                    children = 
                    [
                        html.H1(id='Title1',children='Score',style={'text-align':'center','font-size':'23px','font-color':'white'
                        # ,'background-color': '#000000'
                        }),
                        html.Div([dbc.Tooltip(
                            "The score shows how you have performed on a scale of 100",
                            target="Title1",
                            placement = "auto-end"
                        )]),

                        html.Div(style ={'background-color': '#000000'},
                        children =[
                        
                        
                        html.Div(

                            style={'margin-left':'45px',
                                'color' : '#00EA64',
                                'background-color': '#000000',
                                'margin-top':'20px',
                                'margin-bottom':'20px'},
                            
                            
                            children =[
                            daq.LEDDisplay(
                                id='my-daq-leddisplay',
                                value=score,
                                color = '#00EA64',
                                
                                theme = theme,
                                
                        )]),
                        html.Div(

                        style ={'margin-left':'38px','background-color': '#000000'},
                         
                        children = [

                        daq.Gauge(
                            id='my-daq-gauge',
                            color={"gradient":True,"ranges":{"red":[0,50],"yellow":[50,70],"green":[70,100]}},
                            value=score,
                            # label='Score',
                            max=100,
                            min=0,
                            size = 120,
                            theme = theme
                        )])])])   
                    ,md = 2),   

################### Score end ############################################################   

                # #############  User video layout start ##################################                     
                dbc.Col
                (                      
                    html.Div(children=[
                    html.H1(id='Title5',children='User Video',style={'text-align':'center','font-size':'23px','margin-left':'80px'}),
                    html.Div
                    (
                        style =
                        {
                            'width': '46%',
                            'float': 'left',
                            # 'margin': '0% 5% 1% 5%'
                        },
                        children = 
                        [
                            
                            dash_player.DashPlayer
                            (
                                id = 'video-player',
                               
                                url = "static/StudentminorerrorV4.mP4",
                                controls = False,
                                width = '330%',
                                height = '430%'  
                            ),
                        ]
                    )]),
                    md= 3,
                ),
# #############  User video layout start ################################## 

# #############  Expert video layout start ################################## 
                 dbc.Col
                (                      
                    html.Div(children = [
                    html.H1(id='Title7',children='Expert Video',style={'text-align':'center','font-size':'23px','margin-left':'266px','width':'200px'}),
                    html.Div
                    (
                        style =
                        {
                            'width': '46%',
                            'float': 'left',
                            'margin-left': '140px'
                        },
                        children = 
                        [
                            
                            dash_player.DashPlayer
                            (
                                id = 'video-player2',
                                
                                url = "static/TeacherminorerrorV4.mp4",
                                controls = False,
                                width = '330%',
                                height = '430%'  
                            ),
                        ]
                    ),
                ]),md=3,

                ),

# #############  Expert video layout start end ################################## 

################ Control Start ################################

                 dbc.Col
                (
                    html.Div(

                        style =
                        {
                                # 'width': '92%',
                                # 'float': 'left',
                                'margin-left': '100px'
                        },  
                        
                        children=[

                          
                    
                        html.H1(id='Title6',children='Controls',style={'text-align':'left','font-size':'23px','margin-left': '280px'}),
                        html.Div
                        ( 
                            style =
                            {
                                'width': '70%',
                                # 'float': 'left',
                                'margin-left': '220px'
                            },
                            children = 
                            [

                                   
                                # html.Div
                                # (
                                #     id='div-current-time',
                                #     # style={'margin': '10px 0px'}
                                # ),
                                # html.P("Update Interval for Current Time:"),
                                # html.Div(
                                # style={'margin-bottom': '10px'},
                                # children = [


                                # dcc.Slider(
                                #     id='slider-intervalCurrentTime',
                                #     min=1,
                                #     max=1000,
                                #     step=None,
                                #     updatemode='drag',
                                #     marks={i: str(i) for i in [1,10, 40, 100, 200, 500, 1000]},
                                #     value=10
                                # )]),

                                dcc.Checklist
                                (
                                    id='radio-bool-props',
                                    options=[{'label': val.capitalize(), 'value': val} for val in
                                    [
                                        'playing',
                                            # 'loop',
                                            # 'controls',
                                        'muted'
                                    ]],
                                    value=['muted']

                                ),
                                   

                                html.P("Volume:", 
                                style =
                                {
                                   'margin-top': '10px'
                                }),


                                dcc.Slider
                                (
                                    id = 'slider-volume',
                                    min = 0,
                                    max = 1,
                                    step = 0.05,
                                    value = 1,
                                    updatemode = 'drag',
                                    marks = {
                                        0: '0%',
                                        1: '100%'
                                    }
                                ),

                                html.P("Playback Rate:", 
                                style =
                                {
                                    'margin-top': '25px'
                                }),
                                dcc.Slider
                                (
                                    id = 'slider-playback-rate1',
                                    min = 0,
                                    max = 4,
                                    step = None,
                                    updatemode = 'drag',
                                    marks = {
                                        i: str(i) + 'x' 
                                        for i in [0, 0.25, 0.5, 0.75, 1, 2, 3, 4]
                                    },
                                    value = 1
                                ),

                             ]
                     )
                    ]),md=2.3,
                    
                ),     

               
####################### Controls end ########################                 




            ],
        # no_gutters=True, 
        # className="h-25",
        

            
        ),

        
        dbc.Row
        (
            [

################### Analysis upper and low body start #################################                
                
                dbc.Col
                (
                    
                        html.Div(

                        style =
                        {
                            'margin-left': '40px',
                            'margin-top':'15px',
                            
                        },
                        children=
                        [
                        
                            html.H1(id='Title4',children='Analysis',style={'text-align':'center','font-size':'23px','margin-bottom':'20px'}),
                            # html.Div
                            #     (
                            #         id='river2',
                            #         style={'margin': '10px 0px'}
                            #     ),
                            # html.Div
                            #     (
                            #         id='river3',
                            #         style={'margin': '10px 0px'}
                            #     ),    
                            html.Div([dbc.Tooltip(
                            "user's upper body and lower body performance accuracy with respect to the expert",
                            target="Title4",
                            placement = "top"
                            )]),
                            html.Div
                                (
                                    id='div-current-time',
                                    # style={'margin': '10px 0px'}
                                ),
                            html.Div(
                                style =
                                {
                                    'display': 'flex',
                                    'justify-content':'space-evenly',
                                    'color' : '#00EA64',
                                    # 'background-color': '#000000'
                                    
                                    
                            
                            
                                },
                                children=
                                [
                                html.Div(
                                style =
                                {
                                 'margin-right':'20px' , 
                                },
                                children = [        

                                daq.Tank
                                (
                                id='my-daq-tank2',
                                min=0,
                                max=100,
                                width=30,
                                # height=140,
                                # style={'background-color': '#000000'},

                                label="Upper body",
                                value=upperbodyscore,
                                color = '#00EA64',
                                scale={
                                    "custom": {

                                
                                     
                                    '20': {"style": {"color": '#00EA64'}, "label": "20"},
                                    '40': {"style": {"color": '#00EA64'}, "label": "40"},
                                    '60': {"style": {"color": '#00EA64'}, "label": "60"},
                                    '80': {"style": {"color": '#00EA64'}, "label": "80"},
                                    '100': {"style": {"color": '#00EA64'}, "label": "100"},
                                    
                                    } 
                                
                                },
                                
                                # color = '#00EA64',
                                                                
                                )]),
                            daq.Tank(
                                id='my-daq-tank1',
                                min=0,
                                max=100,
                                width=30,
                                label={
                                    'label': 'Lower Body',
                                    'color': '#00EA64',
                                },    
                                value=lowerbodyscore,
                                color = '#00EA64',
                                
                                scale={
                                    "custom": {

                                
                                     
                                    '20': {"style": {"color": '#00EA64'}, "label": "20"},
                                    '40': {"style": {"color": '#00EA64'}, "label": "40"},
                                    '60': {"style": {"color": '#00EA64'}, "label": "60"},
                                    '80': {"style": {"color": '#00EA64'}, "label": "80"},
                                    '100': {"style": {"color": '#00EA64'}, "label": "100"},
                                    
                                    } 
                                
                                },
                                
                                
                                
                                
                                
                            )
                            ])
                            
                        ]),
                    md=2.8,
                ),

################### Analysis upper and low body end #################################     
               
#############  Main Graph layout start ##################################
                
                
                dbc.Col(
                    
                    style =
                            {
                                'height' : '40px',
                                'justify-content': 'center',
                                'background-color': '#000000',
                                'margin-left':'50px',
                            
                            
                            },
                    
                    children =[
                    
                                        
                    html.Div(

                        
                        
                        children=[
                        
                        html.H1(id='Title2',children='Main Graph',style={'text-align':'center','font-size':'23px'}),
                        html.Div([dbc.Tooltip(
                            "The graph (error vs time)shows the places where you need to improve.The higher the peak, higher the error",
                            target="Title2",
                            placement = "auto"
                        )]),

                        html.Div(

                        style =
                            {
                                # 'height' : '100px',
                                'justify-content': 'center',
                                'background-color': '#000000',
                            
                            
                            },
                        children=[
                        dcc.RadioItems(
                            id="leftright-radio",
                            
                            options=[
                                {"label": k, "value": k}
                                for k in all_options.keys()
                            ],
                            value="difference"),
                        dcc.RadioItems(
                            id="bodypart-radio", 
                            value="difference"
                        )]),    


                        dcc.Graph(id="the_graph",config={'displayModeBar': False,'editable': True,'edits': {'shapePosition': True}}
                        )]),
                    
                              
                                   
                    
                       
                    
                    ],md=4.5),

#############  Main Graph layout end ##################################

# #############  Error Graph layout start ##################################                   

                dbc.Col
                (


                       
                
                    html.Div(
                    style =
                            {
                                
                                'margin-left': '20px',
                            
                            
                            },    
                    children=
                    [

                        html.H1(id='Title3',children='Error Graph',style={'text-align':'center','font-size':'23px'}),                                      

                        html.Div([dbc.Tooltip(
                            "The graph shows the error of different body parts at that particular timestamp clicked on main graph.",
                            target="Title3",
                            placement = "auto"
                        )]),
                            
                        
                        
                        
                        dcc.Graph(id="graph1",config={'displayModeBar': False}),
                    ]),md=3
                ),
                      
# #############  Error Graph layout end ################################## 
               
        ],
        # no_gutters=True, 
        # className="h-50",
        
    ),
    # style={"height": "100vh", "width": "90%",    "margin": "0 auto"},
]),
])

    
    
    # fluid=True,

                    
                
                    
                           
        
          # 
        

@app.callback(Output('div-current-time', 'children'),
              [Input('video-player', 'currentTime')])
def update_time(currentTime):
    return 'Current Time: {}'.format(currentTime)

# # @app.callback(Output('river2', 'children'),
# #               [Input('video-player2', 'currentTime')])
# # def update_time(currentTime):
# #     return 'Current Time: {}'.format(currentTime)

# @app.callback(Output('river3', 'children'),
#               [Input('video-player', 'currentTime')])
# def update_time(currentTime):
#     return currentTime 





@app.callback(
    Output('bodypart-radio', 'options'),
    Input('leftright-radio', 'value'))
def set_bodypart_options(selected_leftright1):
    # print(all_options)
    print(selected_leftright1)
    if selected_leftright1:
        return [{'label': i, 'value': i} for i in all_options[selected_leftright1]]


@app.callback(
    Output('the_graph','figure'),
    Input('leftright-radio', 'value'),
    Input('bodypart-radio', 'value'),
    Input('the_graph', 'clickData'),
    Input('the_graph', 'selectedData'),
    Input('video-player','currentTime')
    
    )

def update_figure1(selected_leftright, selected_bodypart,clickData,selectedData,z):

   
    figure = go.Figure()

    # figure.add_trace(go.Scatter(x=[0,13], y=[53, 53], 
    #                 mode='lines',
    #                 opacity = 0.005,
    #                 line=dict(width=0.5, color='rgba(153,213,148,0.05)',
    #                 # stackgroup='one'
    #                 )))   

    # figure.add_trace(go.Scatter(x=[0,13], y=[53, 70], 
    #                 mode='lines',
    #                 opacity = 0.005,
    #                 line=dict(width=0.5, color='rgba(255,255,191,0.05)',
    #                 # stackgroup='one'
    #                 ))) 

    # figure.add_trace(go.Scatter(x=[0,13], y=[70, 100], 
    #                 mode='lines',
    #                 opacity = 0.005,
    #                 line=dict(width=0.5, color='rgba(252,141,89,0.05)',
    #                 # stackgroup='one'
    #                 )))                                       
    
#######    

    figure.add_trace(go.Scatter(
    x=[0,14], y=[53, 53],
    mode='lines',
    line=dict(width=0.5, color='rgb(111, 231, 219)'),
    stackgroup='one',name="very low Error"
    ))
    figure.add_trace(go.Scatter(
    x=[0,14], y=[17, 17],
    mode='lines',
    line=dict(width=0.5, color='rgb(255,255,191)'),
    stackgroup='one',name="Medium Error"
    ))
    figure.add_trace(go.Scatter(
    x=[0,14], y=[30, 30],
    mode='lines',
    line=dict(width=0.5, color='rgb(252,141,89)'),
    stackgroup='one',name="High Error"
    ))
    
    

    if (selected_bodypart == 'difference'):
        
        figure.add_trace((go.Scatter(x=df7['time'], y=df7['diffnormal'],name = 'difference',mode = 'markers+lines',
            marker={ 'color': 'rgb(0,0,120)','size' : 1},showlegend=False)))

        
        

    if (selected_bodypart == 'leftlegdiff'):
        figure.add_trace((go.Scatter(x=df7['time'], y=df7['leftlegdiffnormal'],name = 'leftlegdifference',mode = 'markers+lines',
            marker={ 'color': 'rgba(0, 116, 217, 0.7)','size' : 2},showlegend=False)))


    if (selected_bodypart == 'rightlegdiff'):
        figure.add_trace((go.Scatter(x=df7['time'], y=df7['rightlegdiffnormal'],name = 'rightlegdifference',mode = 'markers+lines',
            marker={ 'color': 'rgba(0, 116, 217, 0.7)','size' : 2},showlegend=False)))

    if (selected_bodypart == 'leftarmdiff'):
        figure.add_trace((go.Scatter(x=df7['time'], y=df7['leftarmdiffnormal'],name = 'leftarmdifference',mode = 'markers+lines',
            marker={ 'color': 'rgba(0, 116, 217, 0.7)','size' : 2},showlegend=False)))


    if (selected_bodypart == 'rightarmdiff'):
        figure.add_trace((go.Scatter(x=df7['time'], y=df7['rightarmdiffnormal'],name = 'rightarmdifference',mode = 'markers+lines',
            marker={ 'color': 'rgba(0, 116, 217, 0.7)','size' : 2},showlegend=False)))


    if (selected_bodypart == 'leftshoulderdiff'):
        figure.add_trace((go.Scatter(x=df7['time'], y=df7['leftshoulderdiffnormal'],name = 'leftshoulderdifference',mode = 'markers+lines',
            marker={ 'color': 'rgba(0, 116, 217, 0.7)','size' : 2},showlegend=False)))          

    
    if (selected_bodypart == 'rightshoulderdiff'):
        figure.add_trace((go.Scatter(x=df7['time'], y=df7['rightshoulderdiffnormal'],name = 'rightshoulderdifference',mode = 'markers+lines',
            marker={ 'color': 'rgba(0, 116, 217, 0.7)','size' : 2},showlegend=False)))

  
    

    # figure.update_layout(
    #         yaxis=dict
    #         (

    #             tickformat = '.3f',
                
                               
    #         )         
    #     )     

    figure.update_layout(
        xaxis_title="Time",
        yaxis_title="Difference",
        template="plotly_dark"
        )  

    figure.update_layout(height = 300, width = 810) 

    # figure.add_trace(go.Scatter(x=[0,13], y=[50, 50], 
    #                 mode='lines',
    #                 opacity = 0.05,
    #                 line=dict(width=0.5, color='lightgreen'),
    #                 stackgroup='one'
    #                 ))         
    # figure.update_layout(modebar_remove = ['zoom'])
    
    # figure.update_yaxes(range=[0, 100],tick0=0, dtick=20,showgrid= True)
    # figure.update_xaxes(tick0=0, dtick=1,showgrid= False)
    figure.update_layout(clickmode='event+select')
    if selectedData is not None:
        print("selected")
        print(selectedData['points'])
    if clickData is not None:
        print('clicked')
        print(clickData['points'])
        selectedData =  clickData
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']
           
        figure.add_trace(go.Scatter( mode='markers',
        x=[x],
        y=[y],
        marker=dict(
        color='LightSkyBlue',
        size=3.5,
        opacity=0.9,
        line=dict(
        color='#d62728',
        width=8
        )
        ),
        showlegend=False
        )
        )

    
    figure.add_shape(type="line",
    x0=z, y0=0, x1=z, y1=80,xref= 'x',yref='y',
    line=dict(color="RoyalBlue",width=3)
    )


    # figure.add_trace(go.Scatter(x=[0,13], y=[50, 50], fill='#91cf60',
    #                 mode='none' # override default markers+lines
    #                 ))

    # figure.add_trace(go.Scatter(x=[0,13], y=[50, 50], 
    #                 mode='lines',
    #                 opacity = 0.05,
    #                 line=dict(width=0.5, color='lightgreen'),
    #                 stackgroup='one'
    #                 ))                

    


    return figure


# @app.callback(Output('video-player', 'intervalCurrentTime'),
#               [Input('slider-intervalCurrentTime', 'value')])
# def update_intervalCurrentTime(value):
#     return value



@app.callback(Output('video-player', 'playing'),
              Output('video-player2', 'playing'),
              [Input('radio-bool-props', 'value')])
def update_prop_playing(values):
    print("the value is none")
    if values is not None:
        print("the value is")
        print(values)
        return 'playing' in values,'playing' in values


@app.callback(Output('video-player', 'muted'),
              Output('video-player2', 'muted'),
              [Input('radio-bool-props', 'value')])
def update_prop_playing2(values):
    print("the value is none")
    if values is not None:
        print("the value is")
        print(values)
        return 'muted' in values,'muted' in values

      





@app.callback(Output('video-player', 'volume'),
            Output('video-player2', 'volume'),
              [Input('slider-volume', 'value')])
def update_volume(value):
    return value,value


@app.callback(Output('video-player', 'playbackRate'),
              Output('video-player2', 'playbackRate'),
              [Input('slider-playback-rate1', 'value')])
def update_playbackRate(value):
    return value,value 

# @app.callback(
#     Output('relayout-data', 'children'),
#     [Input('the_graph', 'relayoutData')])
# def display_selected_data(relayoutData):
#     print(json.dumps(relayoutData, indent=2))
#     return json.dumps(relayoutData, indent=2)


# @app.callback(
#     Output('video-player', 'seekTo'),
#     Output('video-player','currentTime'),
#     Output('video-player2', 'seekTo'),
#     Output('video-player2','currentTime'),
#     [Input('the_graph', 'relayoutData')])
# def display_selected_data2(relayoutData):
#     print(json.dumps(relayoutData, indent=2))
    
# @app.callback(
#     Output('the_graph', 'relayoutData'),
#     [Input('the_graph', 'relayoutData')])
# def display_selected_data(relayoutData):
#     print(json.dumps(relayoutData, indent=2))
#     return json.dumps(relayoutData, indent=2)    

  

 
    

@app.callback(
    Output('graph1', 'figure'),
    Output('video-player', 'seekTo'),
    Output('video-player','currentTime'),
    Output('video-player2', 'seekTo'),
    Output('video-player2','currentTime'),
    
    [Input('the_graph', 'clickData')])
def display_click_data(clickData):
    print(json.dumps(clickData))
    figure1 = go.Figure()
    

    if (clickData is None):
        z = 0
        figure1.update_layout(
            xaxis_title="Time",
            yaxis_title="Difference",
            template="plotly_dark"
        )  
        figure1.update_layout(height = 330, width = 425 )
        return figure1,z,z,z,z

    else:    
        z = clickData['points'][0]['x']
        print (z)
        filtered_df1 = df7[df7.time == z]
        filtered_df = filtered_df1[["time","diffnormal","hipsdiff","leftlegdiffnormal","rightlegdiffnormal","leftlegupdiffnormal","rightlegupdiffnormal","leftarmdiffnormal","leftforearmdiffnormal","rightarmdiffnormal","rightforearmdiffnormal","leftshoulderdiffnormal","rightshoulderdiffnormal"]]
    
        # figure1 = go.Figure()

        #######    

     
    
         
        
        figure1.add_trace(go.Bar(x=filtered_df['time'], y=filtered_df['leftshoulderdiffnormal'],name = 'leftshouler',marker_color = 'LightSkyBlue'))
        figure1.add_trace(go.Bar(x=filtered_df['time'], y=filtered_df['rightshoulderdiffnormal'],name = 'rightshoulder',marker_color = 'Mediumblue'))

        figure1.add_trace(go.Bar(x=filtered_df['time'], y=filtered_df['leftarmdiffnormal'],name = 'leftarm',marker_color = 'lightgreen'))
        figure1.add_trace(go.Bar(x=filtered_df['time'], y=filtered_df['rightarmdiffnormal'],name = 'rightarm',marker_color = 'green'))

        figure1.add_trace(go.Bar(x=filtered_df['time'], y=filtered_df['leftlegdiffnormal'],name = 'leftleg',marker_color = 'violet'))  
        figure1.add_trace(go.Bar(x=filtered_df['time'], y=filtered_df['rightlegdiffnormal'],name = 'rightleg',marker_color = 'darkviolet'))
        
        
        figure1.update_layout(height = 330, width = 425 ) 
        figure1.update_layout(
            xaxis_title="Time",
            yaxis_title="Difference", 
            template="plotly_dark"
        )  
        figure1.update_yaxes(range=[0, 100])
        figure1.update_xaxes(tickvals=[z])

        # figure1.add_trace(go.Scatter(
        # x=[0,13], y=[53, 53],
        # mode='lines',
        # line=dict(width=0.5, color='rgb(111, 231, 219)'),
        # stackgroup='one',name="very low Error"
        # ))
        # figure1.add_trace(go.Scatter(
        # x=[0,13], y=[17, 17],
        # mode='lines',
        # line=dict(width=0.5, color='rgb(255,255,191)'),
        # stackgroup='one',name="Medium Error"
        # ))
        # figure1.add_trace(go.Scatter(
        # x=[0,13], y=[30, 30],
        # mode='lines',
        # line=dict(width=0.5, color='rgb(252,141,89)'),
        # stackgroup='one',name="High Error"
        # ))
        
        return figure1,z,z,z,z
        
    
    



if __name__ == "__main__":
    app.run_server(debug=True, port=8060)


