from __future__ import print_function
import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row,SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import NGram
import re
import os
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

warehouseLocation = 'C:\DSP_Project2\days-project2'
#sc = SparkContext(conf=SparkConf().setAppName("MalwareClassifier"))
spark = SparkSession\
        .builder\
        .appName("MalwareDetection")\
        .config('spark.sql.warehouse.dir',warehouseLocation)\
        .getOrCreate()
sc=spark.sparkContext;


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
    byteFileCollect = sc.wholeTextFiles(filePath, 50)
    #byteFileCollect = sc.wholeTextFiles("s3n://eds-uga-csci8360/data/project2/binaries/c2hn9edSNJKmw0OukrBv.bytes")

    # ======
    # Use the below line to test data of byte file
    # byteFileCollect= sc.wholeTextFiles("s3n://eds-uga-csci8360/data/project2/binaries/c2hn9edSNJKmw0OukrBv.bytes",50)
    # ======
    cleanFile = byteFileCollect.map(lambda doc: (doc[0].encode('utf-8'), cleanDoc(doc[1]))).cache()
    wholeTextFileNameRDD = cleanFile.map(lambda (x, y): (os.path.splitext(os.path.basename(x))[0], y))
    #cleanFile.saveAsTextFile("C:\\Users\Shubhi\Desktop\cleanFile.txt")

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
    finalDataSetRDD= dataSet.map(lambda (x,y): (x,y[0],y[1]))
    
    # =========================================================================
    # 1. creating DATAFRAME
    # 2. Generating NGRAMS
    # =========================================================================
    schemaString = "HashCodeFile Label Content"
    fields = [StructField("HashCodeFile", StringType(), True),StructField("Label", StringType(), True),StructField("Content",ArrayType(StringType(),False), True) ]
    schema = StructType(fields)
    schemaByte = spark.createDataFrame(finalDataSetRDD,schema)
    schemaByte.createOrReplaceTempView("byteDataFrame")
    ngram = NGram(n=4, inputCol="Content", outputCol="ngrams")
    ngramDataFrame = ngram.transform(schemaByte)
    #ngramDataFrame.select("ngrams").show(truncate=False)   \\No Use
    #ngramDataFrame.show()

    # =========================================================================
    # RANDOM FOREST
    # =========================================================================
    
    # labelIndexer = StringIndexer(inputCol="Label", outputCol="indexedLabel").fit(schemaByte)
    # # Automatically identify categorical features, and index them.
    # # Set maxCategories so features with > 4 distinct values are treated as continuous.
    # featureIndexer = VectorIndexer(inputCol="Content", outputCol="indexedFeatures", maxCategories=9).fit(schemaByte)

    # (trainingData, testData) = schemaByte.randomSplit([0.7, 0.3])

    # # Train a RandomForest model.
    # rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # # Chain indexers and forest in a Pipeline
    # pipeline = Pipeline(stages=[labelIndexer, featureIndexer, rf])

    # # Train model.  This also runs the indexers.
    # model = pipeline.fit(trainingData)

    # # Make predictions.
    # predictions = model.transform(testData)

    # # Select example rows to display.
    # predictions.select("prediction", "indexedLabel", "features").show(5)

    # # Select (prediction, true label) and compute test error
    # evaluator = MulticlassClassificationEvaluator(
    #     labelCol="indexedLabel", predictionCol="prediction", metricName="precision")
    # accuracy = evaluator.evaluate(predictions)
    # print("Test Error = %g" % (1.0 - accuracy))

    # rfModel = model.stages[2]
    # print(rfModel)  # summary only

if __name__ == "__main__":
    main()
