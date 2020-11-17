from sklearn.feature_extraction import DictVectorizer
import csv
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from pydotplus import graph_from_dot_data


#数据预处理 将密度和含糖率成为离散的属性
df= pd.read_csv('data3.0.csv')
print('数据的尺寸',df.shape)
print('数据的属性',df.head())
print("________")
df["密度"] = (df["密度"]<np.mean(df["密度"])).astype(np.int32)
df["密度"].replace(0,'密度高',inplace=True)
df["密度"].replace(1,'密度低',inplace=True)


df["含糖率"]=(df["含糖率"]<np.mean(df["含糖率"])).astype(np.int32)

df["含糖率"].replace(0,'含糖高',inplace=True)
df["含糖率"].replace(1,'含糖低',inplace=True)

#重新生成新的CSV文件
print(df)
df.to_csv("after_pro_data.csv",index=False,sep=',')

#读取属性
data =open('after_pro_data.csv')
reader = csv.reader(data)#采用csv.reader读取文件
for row in reader:#reader不能直接使用，需要通过循环提取每一行的数据
    headers=row
    break#只需要将属性提取出来

#生成属性表和标签表
featureList = []
labelList = []
for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]
    featureList.append(rowDict)

print("属性表",featureList)
print("标签表",labelList)

#将属性装换成可计算的数据
vec = DictVectorizer()  #实例化
dummyX = vec.fit_transform(featureList) .toarray()
print(vec.get_feature_names())
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)

X_train,X_test,Y_train,Y_test = train_test_split(dummyX,dummyY,test_size=5,random_state=0)

#采用基尼指数最小的属性进行分类

test = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(max_depth=i+1)
    clf = clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    test.append(score)
max_depth=test.index(max(test))
plt.plot(range(1,11),test,color="red")
plt.legend(["max_depth"])
plt.show()

#得到最佳的max_depth来创建决策树
clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=max_depth+1)
clf = clf.fit(X_train, Y_train)


#输出预测结果
predictedY = clf.predict(X_test)
print("predictedY: " + str(predictedY))
#输出正确率
right_rate=clf.score(X_test,Y_test)
print("right_rate:"+str(right_rate))

#生成决策树pdf
feature_name = ['含糖率=含糖低', '含糖率=含糖高', '密度=含糖高', '密度=密度高', '敲声=沉闷', '敲声=浊响', '敲声=清脆', '根蒂=硬挺', '根蒂=稍蜷', '根蒂=蜷缩', '纹理=模糊', '纹理=清晰', '纹理=稍糊', '脐部=凹陷', '脐部=平坦', '脐部=稍凹', '色泽=乌黑', '色泽=浅白', '色泽=青绿', '触感=硬滑', '触感=软粘']

class_name = ["好瓜","坏瓜"]
data=tree.export_graphviz(clf
                     ,feature_names=feature_name #特征
                     ,class_names=class_name    #类别
                     ,filled=True     #颜色填充
                     ,rounded=True    #让边框变得圆润
                    )
graph = graph_from_dot_data(data)
graph.write_pdf('tree.pdf')
