import urllib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, Row


sc = SparkContext(conf = SparkConf().setAppName("Read Data"))


def main():

	sc._jsc.hadoopConfiguration().set('fs.s3n.awsAccessKeyId','AKIAI6K6FXFJLKCKQBWA')
	sc._jsc.hadoopConfiguration().set('fs.s3n.awsSecretAccessKey','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')
	
	hashFileData= sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/X_train_small.txt").map(lambda doc:doc.encode("utf-8").strip())
	entirehashFileData = hashFileData.zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()

	labelData= sc.textFile("s3n://eds-uga-csci8360/data/project2/labels/y_train_small.txt").map(lambda doc:doc.encode("utf-8").strip())
	entireLabelData = labelData.zipWithIndex().map(lambda doc:(doc[1],doc[0])).cache()
	print entireLabelData.collect()
	#fileAndLabelRDD= hashFileData.join(entireLabelData)
	#print fileAndLabelRDD.collect()
	
	#print entireDocData.collect()
	#

if __name__ == "__main__":
    main()