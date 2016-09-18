from __future__ import print_function
from pyspark.sql import SQLContext, Row,SparkSession
from pyspark.sql.types import *
import re
import os

warehouseLocation = 'C:\DSP_2\days-project2'
spark = SparkSession\
        .builder\
        .appName("MalwareClassification")\
        .config("spark.sql.warehouse.dir",warehouseLocation)\
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
    hashFileTestData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/X_test_small.txt").map(lambda doc: doc.encode("utf-8").strip())
    entirehashFileTestData = hashFileTestData.zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()

    # =========================================================================
    # Reading (hashcode).bytes file from s3:
    # 1. Concatenating .bytes extension to each hashcode obtained.
    # 2. Making RDD a string to pass tp wholeTextFile function.
    # 3. Read bytes file from s3 and stored it in RDD format (Filename, FileData)
    # 4. Doing initial cleaning of the data through function cleanDoc()
    # =========================================================================
    
    byteTestFile = hashFileTestData.map(lambda doc: ("s3n://eds-uga-csci8360/data/project2/binaries/" + doc + ".bytes"))
    testFilePath = byteTestFile.reduce(lambda str1, str2: str1 + "," + str2)
    #byteTestFileCollect = sc.wholeTextFiles(testFilePath, 1000)
    
    # ======
    # Use the below line to test data of byte file
    byteTestFileCollect= sc.wholeTextFiles("s3n://eds-uga-csci8360/data/project2/binaries/8lTjbp3rnwtLh104E57v.bytes,s3n://eds-uga-csci8360/data/project2/binaries/61rOtseYwGE8lUgPRTCi.bytes,s3n://eds-uga-csci8360/data/project2/binaries/Hps7JhzwjlOAkMXKP92n.bytes,s3n://eds-uga-csci8360/data/project2/binaries/jOPwcHh7kyFaB4Yv8SMU.bytes,s3n://eds-uga-csci8360/data/project2/binaries/AULFdkoH9Tl5GIBu1jwp.bytes",50)
    # ======
    cleanTestFile = byteTestFileCollect.map(lambda doc: (doc[0].encode('utf-8'), cleanDoc(doc[1])))
    wholeTestTextFileNameRDD = cleanTestFile.map(lambda (x, y): (os.path.splitext(os.path.basename(x))[0], y))

    labelTestData = sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/y_test_small.txt").map(lambda doc: doc.encode("utf-8").strip())
    entireLabelTestData = labelTestData.zipWithIndex().map(lambda doc: (doc[1], doc[0])).cache()

    hashFileLableTestPair=entirehashFileTestData.join(entireLabelTestData,numPartitions=1000)
    hashFileLableTestRDD=hashFileLableTestPair.values()
    hashFileLableTestRDDPair=hashFileLableTestPair.keyBy(lambda line:line[0]).mapValues(lambda line:line[1])
    dataSetTest = hashFileLableTestRDDPair.join(wholeTestTextFileNameRDD,numPartitions=100)
    finalDataTestSetRDD = dataSetTest.map(lambda (x, y): (x, y[0], y[1]))

    # =========================================================================
    # creating DATAFRAME
    # =========================================================================
    schemaString = "hashcodefile label features"
    fields = [StructField("hashcodefile", StringType(), True),StructField("label", StringType(), True),
              StructField("features", ArrayType(StringType(), False), True)]
    schema = StructType(fields)
    schemaTestByte = spark.createDataFrame(finalDataTestSetRDD, schema)
    schemaTestByte.createOrReplaceTempView("byteDataFrame")
    # =========================================================================
    # Reading and writing to Parquet file file from s3
    # =========================================================================
   
    schemaTestByte.write.parquet("C:\Users\Shubhi\Desktop\cleanTestFile.parquet")
    # testDoc= spark.read.parquet("cleanFile.parquet")
    # print(testDoc.show())
    
if __name__ == "__main__":
    main()