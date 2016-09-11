from __future__ import print_function
import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row,SparkSession
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.types import *
from pyspark.ml.feature import NGram
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.mllib.linalg import SparseVector, VectorUDT
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
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
    byteFileCollect = sc.wholeTextFiles(filePath, 20000)
    #byteFileCollect = sc.wholeTextFiles("s3n://eds-uga-csci8360/data/project2/binaries/hkscPWGaIw0ALpHuNKr8.bytes,s3n://eds-uga-csci8360/data/project2/binaries/G31czXvpnwUfRtdJ4TFs.bytes,s3n://eds-uga-csci8360/data/project2/binaries/dETSCuIZDapLP9AlJ7o6.bytes,s3n://eds-uga-csci8360/data/project2/binaries/F3Zj217CLRxgi0NyHMY4.bytes")
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
    ngram = NGram(n=4, inputCol="content", outputCol="4-grams")
    ngramDataFrame = ngram.transform(schemaByte)
    schemaLabeledPoint = StructType([
        StructField("label", DoubleType(), True),
        StructField("features", VectorUDT(), True)
    ])



    #ngramDataFrame.rdd.saveAsTextFile("file:///home/dharamendra/tr.txt")
    #tempRDD=finalDataSetRDD
    #print(finalDataSetRDD.values().taake(1))
    #==============================================================================
    #
    #Term Frequequency
    #=====================================================
    #tokenizer = Tokenizer(inputCol="4-grams", outputCol="tokens")
    #wordsData = tokenizer.transform(ngramDataFrame)
    #for words_label in wordsData.select("unigram", "label").take(3):
    #    print(words_label)
    hashingTF = HashingTF(inputCol="4-grams", outputCol="features", numFeatures=256)

    featurizedData = hashingTF.transform(ngramDataFrame)
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(featurizedData)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(featurizedData)
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = featurizedData.randomSplit([0.7, 0.3])
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=10)

    # Convert indexed labels back to original labels.
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                                   labels=labelIndexer.labels)

    # Chain indexers and forest in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf, labelConverter])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))

    rfModel = model.stages[2]
    print(rfModel)  # summary only

    #for features_label in featurizedData.select("rawFeatures", "label").take(3):
        #print(features_label)
    #idf = IDF(inputCol="rawFeatures", outputCol="features")
    #idfModel = idf.fit(featurizedData)
    #rescaledData = idfModel.transform(featurizedData)
    #for features_label in rescaledData.select("features", "label").take(3):
     #   print(features_label)
    #============================================================
if __name__ == "__main__":
    main()
