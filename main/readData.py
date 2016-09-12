from __future__ import print_function
import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row,SparkSession
from pyspark.ml.feature import IDF, Tokenizer
from pyspark.mllib.feature import HashingTF,IDF
from pyspark.sql.types import *
from pyspark.ml.feature import NGram
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
import re
import os

warehouseLocation = 'file:///home/dharamendra/PycharmProjects/MalwareDetection'
#sc = SparkContext(conf=SparkConf().setAppName("MalwareClassifier"))
spark = SparkSession\
        .builder\
        .appName("NGramExample")\
        .config('spark.sql.warehouse.dir',warehouseLocation)\
        .getOrCreate()
sc=spark.sparkContext;
sqlContext=SQLContext(sc)


def cleanDoc(bytefileData):
    # Removing unwanted items from the list.
    filteredFile = re.sub("\?|\n|\r", "", bytefileData)

    # Removing line pointers.
    removePointer = [word.encode('utf-8') for word in filteredFile.split() if len(word) < 3]
    return removePointer


def main():
    # =========================================================================
    # Access Key and secret key necessary to read data from Amazon S3
    # =========================================================================
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsAccessKeyId', 'AKIAI6K6FXFJLKCKQBWA')
    sc._jsc.hadoopConfiguration().set('fs.s3n.awsSecretAccessKey', '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')

    # =========================================================================
    # Reading training file from s3
    # =========================================================================
    hashFileData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/X_train_small.txt").map(lambda doc: doc.encode("utf-8").strip())
    entirehashFileData = hashFileData.zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()

    # =========================================================================
    # Reading (hashcode).bytes file from s3:
    # 1. Concatenating .bytes extension to each hashcode obtained.
    # 2. Making RDD a string to pass tp wholeTextFile function.
    # 3. Read bytes file from s3 and stored it in RDD format (Filename, FileData)
    # 4. Doing initial cleaning of the data through function cleanDoc()
    # =========================================================================
    byteFile = hashFileData.map(lambda doc: ("s3n://eds-uga-csci8360/data/project2/binaries/" + doc + ".bytes"))
    filePath = byteFile.reduce(lambda str1, str2: str1 + "," + str2)
    #byteFileCollect = sc.wholeTextFiles(filePath, 20000)
    byteFileCollect = sc.wholeTextFiles("s3n://eds-uga-csci8360/data/project2/binaries/hkscPWGaIw0ALpHuNKr8.bytes,s3n://eds-uga-csci8360/data/project2/binaries/G31czXvpnwUfRtdJ4TFs.bytes,s3n://eds-uga-csci8360/data/project2/binaries/dETSCuIZDapLP9AlJ7o6.bytes,s3n://eds-uga-csci8360/data/project2/binaries/F3Zj217CLRxgi0NyHMY4.bytes,s3n://eds-uga-csci8360/data/project2/binaries/c2hn9edSNJKmw0OukrBv.bytes")
    # ======
    # Use the below line to test data of byte file
    # byteFileCollect= sc.wholeTextFiles("s3n://eds-uga-csci8360/data/project2/binaries/c2hn9edSNJKmw0OukrBv.bytes",50)
    # ======
    cleanFile = byteFileCollect.map(lambda doc: (doc[0].encode('utf-8'), cleanDoc(doc[1]))).cache()
    wholeTextFileNameRDD = cleanFile.map(lambda (x, y): (os.path.splitext(os.path.basename(x))[0], y))


    # cleanFile.saveAsTextFile("C:\\Users\Shubhi\Desktop\cleanFile.txt")

    # =========================================================================
    # Reading label file from s3
    # =========================================================================
    labelData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/y_train_small.txt").map(lambda doc: doc.encode("utf-8").strip()).cache()
    entireLabelData = labelData.zipWithIndex().map(lambda doc: (doc[1], doc[0])).cache()


# =========================================================================
# Joining RDD's of HashFile,Label and content
# =========================================================================
# fileAndLabelRDD= entirehashFileData.join(entireLabelData)
# labelsAndInputJoinedRDD = fileAndLabelRDD.map(lambda (x,y):(x,y[0],y[1]))
    hashFileLablePair=entirehashFileData.join(entireLabelData)
    hashFileLableRDD=hashFileLablePair.values()
    hashFileLableRDDPair=hashFileLableRDD.keyBy(lambda line:line[0]).mapValues(lambda line:line[1])
    dataSet = hashFileLableRDDPair.join(wholeTextFileNameRDD)
    finalDataSetRDD = dataSet.map(lambda (x, y): (x, y[0], y[1]))

    # =========================================================================
    # creating DATAFRAME
    # =========================================================================
    schemaString = "hashcodefile label content"
    fields = [StructField("hashcodefile", StringType(), True), StructField("label", StringType(), True),
              StructField("content", ArrayType(StringType(), False), True)]
    schema = StructType(fields)
    schemaByte = spark.createDataFrame(finalDataSetRDD, schema)
    schemaByte.createOrReplaceTempView("byteDataFrame")
    ngram = NGram(n=2, inputCol="content", outputCol="2-grams")
    ngramDataFrame = ngram.transform(schemaByte).select("hashcodefile","label","2-grams")
    ngramDataFrameString=ngramDataFrame.withColumn("2-gramString",ngramDataFrame["2-grams"].cast(StringType())).select("label","2-gramString")

    ngramDataFrameStringRDD=ngramDataFrameString.rdd
    temp=ngramDataFrameStringRDD.flatMap(lambda line:line[1].split(",")).map(lambda x: (int(str(x.replace(" ", "")).encode('hex'),16), 1)) .reduceByKey(lambda x,y:x+y).map(lambda x:(x[1],x[0]))
    #temp.saveAsTextFile("file:///home/dharamendra/t7.txt")
    # tempDF=temp.toDF("frequency","2-gram")
    fields = [StructField("frequency", LongType(), True), StructField("2-gram", LongType(), True)]
    schemaFreq = StructType(fields)
    tempDF = spark.createDataFrame(temp, schemaFreq)
    tempDF.createOrReplaceTempView("frequencyView")
    count=spark.sql("SELECT SUM(frequency) AS frequencySum from frequencyView")
    #count.rdd.saveAsTextFile("file:///home/dharamendra/t6.txt")
    #==============================================================================
    #
    #Term Frequequency
    #=====================================================
    #
    hashingTF = HashingTF(35468)
    data_hashed = ngramDataFrameStringRDD.map(lambda (label, text): LabeledPoint(label, hashingTF.transform(text)))
    data_hashed.persist()
    #idf = IDF().fit(data_hashed)
    #idfIgnore = IDF(minDocFreq=2).fit(data_hashed)
    #tfidfIgnore = idf.transform(data_hashed)
    #tfidfIgnore.saveAsTextFile("file:///home/dharamendra/t9.txt")
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data_hashed.randomSplit([0.7, 0.3])
    #
    model = RandomForest.trainClassifier(trainingData, numClasses=10, categoricalFeaturesInfo={10:2},
                                          numTrees=5, featureSubsetStrategy="auto",
                                          impurity='gini', maxDepth=7, maxBins=16)
    #
    # # Evaluate model on test instances and compute test error
    predictions = model.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testErr = labelsAndPredictions.filter(lambda (v, p): v != p).count() / float(testData.count())
    print('Test Error = ' + str(testErr))
    print('Learned classification forest model:')
    print(model.toDebugString())
    model.save(sc, "file:///home/dharamendra/myRandomForestClassificationModel")
    sameModel = RandomForestModel.load(sc, "file:///home/dharamendra/myRandomForestClassificationModel")
    #============================================================
if __name__ == "__main__":
    main()
