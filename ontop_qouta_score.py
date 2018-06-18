from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext, SparkSession
from pyspark.sql import SQLContext, HiveContext
import pyspark.sql.functions as func
import pandas as pd
import numpy as np
import re
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark.sql.types import IntegerType
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import when
import logging
from pyspark.sql.functions import count, avg, sum
from pyspark.mllib.linalg.distributed import *
from pyspark.mllib.linalg import Vectors 
from pyspark.sql.functions import monotonically_increasing_id 
from pyspark.sql.window import Window as W
from pyspark.sql.functions import col
from pyspark.sql.functions import *
def extract(row):
    return (row.id, ) + tuple(row.vector.toArray().tolist())
if __name__ == "__main__":
    logging.getLogger("py4j").setLevel(logging.ERROR)
    conf = SparkConf().setAppName("ontop qouta and trotting score")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n###################################################################################################")
    print("Start Calculating Ontop qouta Score")
    print("###################################################################################################")

    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n###################################################################################################")
    print("Start Reading Ontop Preferences Data")
    print("###################################################################################################")
    ontop_preferences = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").option("delimiter", '|').load('/preprocessed_cvm/package_preferences_rowId_Manual')
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n###################################################################################################")
    print("Finished Reading Ontop Preferences Data")
    print("###################################################################################################")

    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n###################################################################################################")
    print("Start Reading Customer Preferences Data")
    print("###################################################################################################")
    customer_persona = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").option("delimiter", '|').load('/preprocessed_cvm/customer_persona_rowId_Manual')
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n###################################################################################################")
    print("Finished Reading Customer Preferences Data")
    print("###################################################################################################")

    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n###################################################################################################")
    print("Start Creating Customer Preferences Block Matrix")
    print("###################################################################################################")
    index_ct = customer_persona.drop("analytic_id")
    index_anaId = customer_persona.select("id","analytic_id")
    index_ct.registerTempTable("index_ct")
    ontop_pref_price = ontop_preferences.select("id","Package_Size_Qouta_S","Package_Size_Qouta_M","Package_Size_Qouta_L","Package_Size_Trotting_S","Package_Size_Trotting_M","Package_Size_Trotting_L","Package_Size_Social","Package_Size_Entertain")
    ontop_pref_price = ontop_pref_price.orderBy(asc("id"))
    bmB_1 = IndexedRowMatrix(ontop_pref_price.rdd.map(lambda x: IndexedRow(x[0], Vectors.dense(x[1:])))).toBlockMatrix(rowsPerBlock=222)
    count = customer_persona.count()
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n###################################################################################################")
    print("Finished Creating Customer Preferences Block Matrix")
    print("###################################################################################################")

    loop = int(count/200000)
    startId = 1
    i = 0
    res = index_ct
    del customer_persona
    print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n###################################################################################################")
    print("Start Matrix Multiplication")
    print("Number of loop: " + str(loop))
    print("###################################################################################################")
    for x in range(loop):
        print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n###################################################################################################")
        print("Starting partial Matrix Multiplication: "+str(x+1) + "/" +str(loop))
        print("###################################################################################################")
        if x != loop-1:
            batch = sqlContext.sql("SELECT * FROM index_ct WHERE id BETWEEN " + str(i+1) + " AND " + str(i+200001))
            matA = IndexedRowMatrix(batch.rdd.map(lambda row: IndexedRow(row[0], Vectors.dense(row[1:]))))
            bmA = matA.toBlockMatrix(colsPerBlock=222)
            bm_Package_Size_384Kbpscore_result = bmA.multiply(bmB_1)
            tmp = bm_Package_Size_384Kbpscore_result.toIndexedRowMatrix().rows.sortBy(lambda x : x.index).map(lambda x:  (x.index,x.vector)).toDF(["id", "vector"])
            tmp = tmp.rdd.map(extract).toDF(["id"])
            tmp = tmp.join(index_anaId, ["id"], "left_outer")
            tmp = tmp.selectExpr("analytic_id", "id", "_2 as Package_Size_Qouta_S", "_3 as Package_Size_Qouta_M","_4 as Package_Size_Qouta_L", "_5 as Package_Size_Trotting_S","_6 as Package_Size_Trotting_M","_7 as Package_Size_Trotting_L","_8 as Package_Size_Social","_9 as Package_Size_Entertain")
            if i == 0:
                #tmp.repartition(1).write.option("sep","|").option("header","true").csv("/ontop_pref/tmp_score_" + str(i))
                res = tmp
            else:
                #tmp.repartition(1).write.option("sep","|").option("header","false").csv("/ontop_pref/tmp_score_" + str(i))
                res = res.unionAll(tmp)
        else:
            batch = sqlContext.sql("SELECT * FROM index_ct WHERE id >= "+ str(i+1))
            matA = IndexedRowMatrix(batch.rdd.map(lambda row: IndexedRow(row[0], Vectors.dense(row[1:]))))
            bmA = matA.toBlockMatrix(colsPerBlock=222)
            bm_Package_Size_384Kbpscore_result = bmA.multiply(bmB_1)
            tmp = bm_Package_Size_384Kbpscore_result.toIndexedRowMatrix().rows.sortBy(lambda x : x.index).map(lambda x:  (x.index,x.vector)).toDF(["id", "vector"])
            tmp = tmp.rdd.map(extract).toDF(["id"])
            tmp = tmp.join(index_anaId, ["id"], "left_outer")
            tmp = tmp.selectExpr("analytic_id", "id", "_2 as Package_Size_Qouta_S", "_3 as Package_Size_Qouta_M","_4 as Package_Size_Qouta_L", "_5 as Package_Size_Trotting_S","_6 as Package_Size_Trotting_M","_7 as Package_Size_Trotting_L","_8 as Package_Size_Social","_9 as Package_Size_Entertain")
            res = res.unionAll(tmp)
            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n###################################################################################################")
            print("Finished Matrix Multiplication")
            print("###################################################################################################")

            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n###################################################################################################")
            print("Start Writting Output file")
            print("###################################################################################################")
            res.write.option("sep","|").option("header","true").csv("/ontop_pref/ontop_qouta_score")
            print("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n###################################################################################################")
            print("Finished Writting Output file")
            print("###################################################################################################")
        i = i+1
    sc.stop()