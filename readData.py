import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row
import re


sc = SparkContext(conf = SparkConf().setAppName("Read Data"))

def cleanDoc(bytefileData):
	
	# Removing unwanted items from the list.
	filteredFile=re.sub("\?|\n|\r","",bytefileData)

	# Removing line pointers.
	removePointer= [word.encode('utf-8') for word in filteredFile.split() if len(word)<3]
	return removePointer


def main():

	#=========================================================================
	# Access Key and secret key necessary to read data from Amazon S3
	#=========================================================================
	sc._jsc.hadoopConfiguration().set('fs.s3n.awsAccessKeyId','AKIAI6K6FXFJLKCKQBWA')
	sc._jsc.hadoopConfiguration().set('fs.s3n.awsSecretAccessKey','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')
	
	#=========================================================================
	# Reading training file from s3
	#=========================================================================
	hashFileData= sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/X_train_small.txt").map(lambda doc:doc.encode("utf-8").strip()).cache()
	#entirehashFileData = hashFileData.zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()

	#=========================================================================
	# Reading (hashcode).bytes file from s3:
	# 1. Concatenating .bytes extension to each hashcode obtained.
	# 2. Making RDD a string to pass tp wholeTextFile function.
	# 3. Read bytes file from s3 and stored it in RDD format (Filename, FileData)
	# 4. Doing initial cleaning of the data through function cleanDoc()
	#=========================================================================
	byteFile= hashFileData.map(lambda doc: ("s3n://eds-uga-csci8360/data/project2/binaries/"+doc+".bytes"))
	filePath=byteFile.reduce(lambda str1,str2 : str1 + "," + str2)
   	byteFileCollect= sc.wholeTextFiles(filePath,50)
   	#======
   	# Use the below line to test data of byte file
   	#byteFileCollect= sc.wholeTextFiles("s3n://eds-uga-csci8360/data/project2/binaries/c2hn9edSNJKmw0OukrBv.bytes",50)
   	#======
	cleanFile= byteFileCollect.map(lambda doc:(doc[0].encode('utf-8'),cleanDoc(doc[1]))).cache()
	
	
   	cleanFile.saveAsTextFile("C:\\Users\Shubhi\Desktop\clenFile.txt")
	
	# labelData= sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/y_train_small.txt").map(lambda doc:doc.encode("utf-8").strip()).cache()
	# entireLabelData = labelData.zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()
	
	# fileAndLabelRDD= entirehashFileData.join(entireLabelData)
	# labelsAndInputJoinedRDD = fileAndLabelRDD.map(lambda (x,y):(x,y[0],y[1]))

	
	

if __name__ == "__main__":
    main()