
#-*- coding:utf-8 -*-

import _init_paths
import os
import sys
import time
import cv2
import caffe
import numpy
import dataBase
from sklearn import svm
from sklearn import metrics
import xlwt
from sklearn import preprocessing
from sklearn.decomposition import PCA


#=========================caffe接口======================
#========================================================

def initcaffe(protourl, modelurl):
    caffe.set_device(0)
    caffe.set_mode_gpu()
    net = caffe.Net(protourl, modelurl,caffe.TEST)
    return net




#=======================模型均值===========================
#readmeanfile
#makeplaces205meandata: 生成pl205数据集的meanfile
#====================================================
def readmeanfile(meanfile):
    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(meanfile, 'rb').read())
    # 将均值blob转为numpy.array
    mean_npy = caffe.io.blobproto_to_array(mean_blob)

    return mean_npy[0][:, 16:240, 16:240]

def makeplaces205meandata():
    out = numpy.zeros([3,224,224])
    out[0,:,:] = 105.487823486
    out[1,:,:] = 113.741088867
    out[2,:,:] = 116.060394287
    return out

#=================特征计算===============================================================
#calture_normal_feature: 计算256尺寸下 center crop特征
#calture_pool_feature:   计算设定尺寸下 以一定步长截取部分图片的特征 然后maxpool，和meanpool后结果
#get_feature_scala_256:  计算整个数据库normal_feature
#get_pool_feature：      计算整个数据库pool_feature
#========================================================================================
def calture_normal_feature(pic, caffenet, meanfile):
    sp = pic.shape
    y = sp[0]
    x = sp[1]
    if x<y:
        y = (int)((y*256)/x)
        x = 256
    else:
        x = (int)((x*256)/y)
        y = 256

    im_256 = cv2.resize(pic, (x,y))
    im_224 = im_256[((int)(y/2)-112):((int)(y/2)+112), ((int)(x/2)-112):((int)(x/2)+112)]
    im = numpy.transpose(im_224, (2, 0, 1))
    im = im - meanfile
    caffenet.blobs['data'].data[...] = im
    caffenet.forward()
    feature = caffenet.blobs['fc7'].data[0]

    return feature

def calture_pool_feature(pic, picsize, cropsize, steplength, caffenet, meanfile, parallelnum):
    feature_max= numpy.zeros((parallelnum, 4096))
    feature_mean= numpy.zeros((parallelnum, 4096))
    feature_max = feature_max - 9999
    #3 对每个尺度框图并提取特征
    step = (picsize-cropsize)//steplength
    for m in range(step+1):
        for n in range(step+1):
            x = m*steplength
            y = n*steplength
            if x > picsize-cropsize:
                x = picsize-cropsize
            if y > picsize-cropsize:
                y = picsize-cropsize
            crop = pic[:, y:y+cropsize, x:x+cropsize, :]    ### crop是四维的数组
            im = numpy.transpose(crop, (0, 3, 1, 2))
            im = im - meanfile
            caffenet.blobs['data'].data[...] = im
            caffenet.forward()
            tmp = caffenet.blobs['fc7'].data
            #4 将特征pooling
            #===================================================================
            #这段代码什么意思？
            for i in range(parallelnum):
                for j in range(4096):
                    if tmp[i][j] >= feature_max[i][j]:
                        feature_max[i][j] = tmp[i][j]
                    #tmp[i][j] = tmp[i][j]/(step+1)*(step+1)
                    #feature_mean[i][j] = feature_mean[i][j] + tmp[i][j]

    return feature_max, feature_mean

def get_feature_scala_256(db, cursor, caffenet, tbname, rownum, datafloder, meanfile, featurename):
    for i in range(rownum):
        print '============current id is :%d ==============\r'%(i+1),
        sql = "SELECT URL FROM " + tbname + " WHERE ID = '%d'" % (i+1)
        cursor.execute(sql)
        result = cursor.fetchall()
        if result[0][0][0] != '/':
            url = datafloder+'/'+result[0][0]
        else:
            url = datafloder+result[0][0]
        im_ori = cv2.imread(url)
        feature = calture_normal_feature(im_ori, caffenet, meanfile)
        #==============================================================
        print feature.shape
        #写入数据库
        dataBase.wirte_feature_to_db(db, cursor, feature, tbname, i+1, featurename)

def get_pool_feature(db, cursor, tbname, rownum, picsize, cropsize, steplength, caffenet, datafloder, meanfile,
                         featurename, parallelnum):
        for i in range(int(rownum / parallelnum)):
            print '============current id is :%d ==============\r' % (i * parallelnum + 1)
            starttime = time.time()
            sql = "SELECT URL FROM " + tbname + " WHERE ID >= '%d' and ID <= '%d'" % (
            i * parallelnum + 1, (i + 1) * parallelnum)
            cursor.execute(sql)
            result = cursor.fetchall()
            #tmp1 = time.time() - starttime
            #print tmp1
            im = numpy.zeros((parallelnum, picsize, picsize, 3))
            for j in range(parallelnum):
                if result[j][0][0] != '/':
                    url = datafloder + '/' + result[j][0]
                else:
                    url = datafloder + result[j][0]
                im_ori = cv2.imread(url)
                im[j, :, :, :] = cv2.resize(im_ori, (picsize, picsize))
            feature_max, feature_mean = calture_pool_feature(im, picsize, cropsize, steplength, caffenet, meanfile,parallelnum)
            #tmp2 = time.time() - starttime - tmp1
            #print tmp2
            # 写入数据库
            for j in range(parallelnum):
                dataBase.wirte_feature_to_db(db, cursor, feature_max[j], tbname, i * parallelnum + 1 + j, featurename)
            #tmp3 = time.time() - starttime - tmp1 - tmp2
            #print tmp3

#=====================主成份分析（PCA）=====================
#==========================================================
def process_pca(data, components=4096):
    pca = PCA(n_components=components)
    newData = pca.fit_transform(data)
    return newData

def process_pca2(train_data, test_data, components=4096):   ###wangchen PCA
    pca=PCA(n_components = components)
    scaler = preprocessing.StandardScaler().fit(train_data)
    scalerData1 = scaler.transform(train_data)
    scalerData2 = scaler.transform(test_data)
    pca.fit(scalerData1)
    newData1 = pca.transform(scalerData1)
    newData2 = pca.transform(scalerData2)
    newData = numpy.concatenate((newData1,newData2),0)
    return newData

def process_pca3(train_data, test_data, components=4096):   ###wangchen PCA
    pca=PCA(n_components = components)
    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)
    pca.fit(train_data)
    newData1 = pca.transform(train_data)
    newData2 = pca.transform(test_data)
    newData = numpy.concatenate((newData1,newData2),0)
    return newData

def process_pca4(train_data, test_data, components=4096):   ###wangchen PCA
    pca=PCA(n_components = components)
    pca.fit(train_data)
    newData1 = pca.transform(train_data)
    newData2 = pca.transform(test_data)
    newData = numpy.concatenate((newData1,newData2),0)
    return newData

def process_pca5(train_data, test_data, components=4096):   ###wangchen PCA
    pca=PCA(n_components = components)
    scaler = preprocessing.StandardScaler().fit(train_data)
    scalerData1 = scaler.transform(train_data)
    scalerData2 = scaler.transform(test_data)
    pca.fit(train_data)
    newData1 = pca.transform(scalerData1)
    newData2 = pca.transform(scalerData2)
    newData = numpy.concatenate((newData1,newData2),0)
    return newData


#======================提取特征============================
#==========================================================

def GET_FEATURE1(db, cursor, caffenet, tbname, rownum, datafloder, meanfile, featurename):
    get_feature_scala_256(db, cursor, caffenet, tbname, rownum, datafloder, meanfile, featurename)

def GET_FEATURE2(db, cursor, caffenet, tbname, rownum, datafloder, meanfile, featurename):
    get_feature_scala_256(db, cursor, caffenet, tbname, rownum, datafloder, meanfile, featurename)

def GET_FEATURE3(db,cursor,tbname,rownum,picsize,cropsize,steplength,caffenet,datafloder,meanfile,featurename,parallelnum):
    get_pool_feature(db,cursor,tbname,rownum,picsize,cropsize,steplength,caffenet,datafloder,meanfile,featurename,parallelnum)

def GET_FEATURE4(db,cursor,tbname,rownum,picsize,cropsize,steplength,caffenet,datafloder,meanfile,featurename,parallelnum):
    get_pool_feature(db,cursor,tbname,rownum,picsize,cropsize,steplength,caffenet,datafloder,meanfile,featurename,parallelnum)

def GET_FEATURE5(db,cursor,tbname,rownum,picsize,cropsize,steplength,caffenet,datafloder,meanfile,featurename,parallelnum):
    get_pool_feature(db,cursor,tbname,rownum,picsize,cropsize,steplength,caffenet,datafloder,meanfile,featurename,parallelnum)

def GET_FEATURE8(db, cursor, traintbname, testtbname, trainnum, testnum):
    FEATURE3_train_data, train_label = dataBase.read_feature(db, cursor, traintbname, trainnum, "FEATURE3")
    FEATURE3_test_data, test_label = dataBase.read_feature(db, cursor, testtbname, testnum, "FEATURE3")
    FEATURE4_train_data, train_label = dataBase.read_feature(db, cursor, traintbname, trainnum, "FEATURE4")
    FEATURE4_test_data, test_label = dataBase.read_feature(db, cursor, testtbname, testnum, "FEATURE4")
    FEATURE13_train_data, train_label = dataBase.read_feature(db, cursor, traintbname, trainnum, "FEATURE5")
    FEATURE13_test_data, test_label = dataBase.read_feature(db, cursor, testtbname, testnum, "FEATURE5")
    train_data = numpy.concatenate((FEATURE3_train_data, FEATURE4_train_data, FEATURE13_train_data), 1)     ###wangchen 特征数组拼接
    test_data = numpy.concatenate((FEATURE3_test_data, FEATURE4_test_data, FEATURE13_test_data), 1)
    newData = process_pca2(train_data,test_data,0.99)
    print newData.shape                     ### （6700，4096）
    #写入数据库
    j=0
    for i in range(trainnum):
        dataBase.wirte_feature_to_db(db, cursor, newData[j], traintbname, i+1, "FEATURE8",newData.shape[1])
        j=j+1

    for i in range(testnum):
        dataBase.wirte_feature_to_db(db, cursor, newData[j], testtbname, i+1, "FEATURE8",newData.shape[1])
        j=j+1
    return newData

def GET_FEATURE9(db, cursor, traintbname, testtbname, trainnum, testnum):
    FEATURE3_train_data, train_label = dataBase.read_feature(db, cursor, traintbname, trainnum, "FEATURE3")
    FEATURE3_test_data, test_label = dataBase.read_feature(db, cursor, testtbname, testnum, "FEATURE3")
    FEATURE4_train_data, train_label = dataBase.read_feature(db, cursor, traintbname, trainnum, "FEATURE4")
    FEATURE4_test_data, test_label = dataBase.read_feature(db, cursor, testtbname, testnum, "FEATURE4")
    FEATURE13_train_data, train_label = dataBase.read_feature(db, cursor, traintbname, trainnum, "FEATURE13")
    FEATURE13_test_data, test_label = dataBase.read_feature(db, cursor, testtbname, testnum, "FEATURE13")
    train_data = numpy.concatenate((FEATURE3_train_data, FEATURE4_train_data, FEATURE13_train_data), 1)     ###wangchen 特征数组拼接
    test_data = numpy.concatenate((FEATURE3_test_data, FEATURE4_test_data, FEATURE13_test_data), 1)
    newData = process_pca2(train_data,test_data,0.99)
    print newData.shape                     ### （6700，4096）
    #写入数据库
    j=0
    for i in range(trainnum):
        dataBase.wirte_feature_to_db(db, cursor, newData[j], traintbname, i+1, "FEATURE9",newData.shape[1])
        j=j+1

    for i in range(testnum):
        dataBase.wirte_feature_to_db(db, cursor, newData[j], testtbname, i+1, "FEATURE9",newData.shape[1])
        j=j+1
    return newData

def GET_FEATURE10(db, cursor, traintbname, testtbname, trainnum, testnum):
    FEATURE3_train_data, train_label = dataBase.read_feature(db, cursor, traintbname, trainnum, "FEATURE3")
    FEATURE3_test_data, test_label = dataBase.read_feature(db, cursor, testtbname, testnum, "FEATURE3")
    FEATURE4_train_data, train_label = dataBase.read_feature(db, cursor, traintbname, trainnum, "FEATURE4")
    FEATURE4_test_data, test_label = dataBase.read_feature(db, cursor, testtbname, testnum, "FEATURE4")
    FEATURE13_train_data, train_label = dataBase.read_feature(db, cursor, traintbname, trainnum, "FEATURE13")
    FEATURE13_test_data, test_label = dataBase.read_feature(db, cursor, testtbname, testnum, "FEATURE13")
    train_data = numpy.concatenate((FEATURE3_train_data, FEATURE4_train_data, FEATURE13_train_data), 1)     ###wangchen 特征数组拼接
    test_data = numpy.concatenate((FEATURE3_test_data, FEATURE4_test_data, FEATURE13_test_data), 1)
    newData = process_pca5(train_data,test_data,0.99)
    print newData.shape                     ### （6700，4096）
    #写入数据库
    j=0
    for i in range(trainnum):
        dataBase.wirte_feature_to_db(db, cursor, newData[j], traintbname, i+1, "FEATURE10",newData.shape[1])
        j=j+1

    for i in range(testnum):
        dataBase.wirte_feature_to_db(db, cursor, newData[j], testtbname, i+1, "FEATURE10",newData.shape[1])
        j=j+1
    return newData

def GET_FEATURE13(db,cursor,tbname,rownum,picsize,cropsize,steplength,caffenet,datafloder,meanfile,featurename,parallelnum):
    get_pool_feature(db,cursor,tbname,rownum,picsize,cropsize,steplength,caffenet,datafloder,meanfile,featurename,parallelnum)


#============================SVM 分类================================
#====================================================================

def SVM(train_data, test_data, train_label,test_label, labnum, testnum):
    trainx = numpy.array(train_data)
    testx = numpy.array(test_data)
    print trainx.shape
    clf = svm.SVC(kernel = 'linear') #kernel = 'linear'
    clf.fit(trainx, train_label)
    pred = clf.predict(testx)
    m_precision, m_recall, f1_score = calculate_result(test_label,pred)
    confusematrix,detail = calculate_detail(pred, test_label, labnum, testnum)
    return m_precision, m_recall, f1_score, confusematrix, detail

def fsvm(db, cursor, trainname, testname, trainnum, testnum, labnum, featurename, featurenum=4096):
    print '从数据库取数据...',featurename
    train_data, train_label = dataBase.read_feature(db, cursor, trainname, trainnum, featurename, featurenum)
    test_data, test_label = dataBase.read_feature(db, cursor, testname, testnum, featurename, featurenum)
    print '从数据库取数据完成, 训练SVM并测试...'
    m_precision, m_recall, f1_score, confusematrix, detail =SVM(train_data, test_data, train_label, test_label, labnum, testnum)
    #detailtofile(db, cursor, "./"+testname+"/"+featurename+".txt", detail, testname, test_label)
    #matrixtoexcl(confusematrix, "./"+testname+"/"+featurename+".xls")



#===============结果计算和输出部分=================
#calculate_result 计算正确率 召回率等
#calculate_detail 统计所有错误分类
#detailtofile     将错误分类 URL ID reallabel pre打印到文件
#matrixtoexcl     打印错误分类混淆矩阵
#=================================================
def calculate_result(actual,pred):
    m_precision = metrics.precision_score(actual,pred,average='macro')
    m_recall = metrics.recall_score(actual,pred,average='macro')
    print 'predict info:'
    print 'precision:{0:.3f}'.format(m_precision)
    print 'recall:{0:0.3f}'.format(m_recall)
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred,average='macro'))
    return m_precision, m_recall, metrics.f1_score(actual,pred,average='macro')

def calculate_detail(pred, reallabel, labnum, datanum):
    detail = []
    confusematrix = numpy.zeros([labnum,labnum])
    for i in range(datanum):
        if pred[i] != reallabel[i]:
            tmp = [i+1, reallabel[i], pred[i]]
            detail.append(tmp)
            confusematrix[reallabel[i], pred[i]] += 1
    return confusematrix,detail

def detailtofile(db, cursor, outfileurl, detail, tbname, label):
    f = open(outfileurl, 'a')
    for i in range(len(detail)):
        ID = detail[i][0]
        reallabel =  detail[i][1]
        error = detail[i][2]
        sql = "SELECT URL FROM "+tbname+" WHERE ID = '%d'" % ID
        cursor.execute(sql)
        url = cursor.fetchall()
        f.writelines(['%d' % ID, ' %d' % reallabel,' %d ' % error, '%s' % url[0][0], '\n'])
    f.close()

def probtofile(outfileurl, numpy_data):
    numpy.savetxt(outfileurl, numpy_data, fmt='%2.4f')

def matrixtoexcl(confusematrix, xlsurl):
    wbk = xlwt.Workbook()
    sheet = wbk.add_sheet('sheet 1')
    for i in range(len(confusematrix)):
        for j in range(len(confusematrix)):
            sheet.write(i,j,confusematrix[i,j])
    wbk.save(xlsurl)

