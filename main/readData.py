from __future__ import print_function
import urllib

from pyspark.sql import SQLContext, Row,SparkSession
from pyspark.ml.feature import IDF, Tokenizer,HashingTF
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import *
from pyspark.ml.feature import NGram

import re
import os

warehouseLocation = 'file:///home/dharamendra/PycharmProjects/MalwareDetection'
#sc = SparkContext(conf=SparkConf().setAppName("MalwareClassifier"))
spark = SparkSession\
        .builder\
        .appName("MalwareClassification")\
        .getOrCreate()
sc=spark.sparkContext;
sqlContext=SQLContext(sc)

#######Method for Preprocessing
def cleanDoc(bytefileData):
    # Removing unwanted items from the list.
    filteredFile = re.sub("\?|\n|\r", "", bytefileData)

    # Removing line pointers.
    removePointer = [word.encode('utf-8') for word in filteredFile.split() if len(word) < 3]
    return removePointer

def uniqueByte(inputByte):
    x= list(set(inputByte))
    temp=[]
    for str in x:
        tempStr=str.replace(" ","")
        temp.append(tempStr)
    return temp

########Main method for Ma
def main():

    fieldsDF = [StructField("hashcodefile", StringType(), True), StructField("label", StringType(), True),
                StructField("2-grams", StringType(), False)]
    schemaDF = StructType(fieldsDF)
    schemaByteParque=spark.read.parquet("cleanFile.parquet")
    print ("Parquet File read completed")
    schemaByteParque.createOrReplaceTempView("byteDataFrame")
    ngram = NGram(n=2, inputCol="content", outputCol="2-grams")
    ngramDataFrame = ngram.transform(schemaByteParque).select("hashcodefile","label","2-grams")
    print ("N-gram completed")
    ngramRDD=ngramDataFrame.rdd
    ngramRDDUnique = ngramRDD.map(lambda line: (line[0].encode('utf-8'),line[1], uniqueByte(line[2])))
    print ("N-Gram unique completed")
    #ngramDFUnique=spark.createDataFrame(ngramRDDUnique,schemaDF)
    #ngramDFUnique.show()
    #ngramDataFrameRDDUnique.saveAsTextFile("file:///home/dharamendra/unique.txt");
    ngramDataFrameStringRDD=ngramRDDUnique.map(lambda line:(line[1].encode('utf-8'),' '.join(str(x) for x in line[2])))
    print ("Array type to String Type conversion completed")
    fieldTF = [StructField("label", StringType(), True), StructField("2-grams", StringType(), True)]
    schemaTF = StructType(fieldTF)
    inputTFDF=spark.createDataFrame(ngramDataFrameStringRDD,schemaTF)
    #ngramDataFrameStringRDD.saveAsTextFile("file:///home/dharamendra/unique.txt");

    count=ngramRDDUnique.flatMap(lambda line:(' '.join(str(x) for x in line[2])).split(" ")).distinct().count()
    print ("Count:",count)
    #count.saveAsTextFile("file:///home/dharamendra/unique.txt");

    # #==============================================================================
    # #
    # #Term Frequequency
    # #=====================================================
    # #
    tokenizer = Tokenizer(inputCol="2-grams", outputCol="bytes")
    inputData = tokenizer.transform(inputTFDF)
    hashingTF = HashingTF(inputCol="bytes", outputCol="bytesFeatures", numFeatures=count)
    featurizedData = hashingTF.transform(inputData)
    print ("Term Frequency completed")

    idf = IDF(inputCol="bytesFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    for features_label in rescaledData.select("features", "label").take(3):
         print(features_label)
    ######## Code for Random Forest Classifier
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(rescaledData)

    # Automatically identify categorical features, and index them.
    # Set maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = \
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=5).fit(rescaledData)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = rescaledData.randomSplit([0.7, 0.3])

    # Train a RandomForest model.
    rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", numTrees=5)

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
    print(rfModel)
    #============================================================
if __name__ == "__main__":
    main()
