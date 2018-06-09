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
def multiply_all(t_df):
    for field in t_df.schema.fields:
        name = field.name
        t_df = t_df.withColumn(name, col(name)*col('total_transaction'))
    return t_df

def onehotenc(t_df, column):
    categories = t_df.select(column).distinct().rdd.flatMap(lambda x : x).collect()
    categories.sort()
    for category in categories:
        function = udf(lambda item: 1 if item == category else 0, IntegerType())
        new_column_name = column+'_'+str(category)
        t_df = t_df.withColumn(new_column_name, function(col(column)))
    t_df = t_df.drop(column)
    return t_df
if __name__ == "__main__":
    logging.getLogger("py4j").setLevel(logging.ERROR)
    conf = SparkConf().setAppName("MatrixMul")
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    df_ontop1802 = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").option("delimiter", '|').load('wasb://ds-cvm-hd-rs-devprod-02-2017-09-25t08-15-40-207z@natds201708cvm1sa01.blob.core.windows.net/data_cvm/ASSOCIATION/CVM_PREPAID_RECOMMENDATION_201802.txt')
    master_jean = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").option("delimiter", ',').load('/data_cvm/master_tariff_jean.csv')
    master_jean3 = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").option("delimiter", ',').load('/data_cvm/master_tariff_jean5.csv')
    masterjean3map = master_jean3.select(col("PromotionCode").alias('Promotion Code'),col("PackageName").alias("New_Package_Name"))
    master_mapping = master_jean.select(col("Promotion Code"),col("Package Name")).join(masterjean3map, ["Promotion Code"], "left_outer")
    master_mapping = master_mapping.drop_duplicates(subset=['Package Name'])
    master_mapping = master_mapping.drop("Promotion Code")
    package_preferences = master_jean.drop_duplicates(subset=['Package Name'])
    parsed_PP = package_preferences.select(col('Package Name').alias('Package_Name'),col('Price Inc VAT').alias("Price"),col('MM Data Speed').alias("Package_Size"))
    parsed_PP_duration = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").option("delimiter", ',').load('/data_cvm/parsed_pp_with_duration.csv')
    parsed_PP_duration1 = parsed_PP_duration.withColumn('Package_Duration_New2',when(parsed_PP_duration.Package_Duration <= 1,'XS').otherwise(parsed_PP_duration.Package_Duration)).drop(parsed_PP_duration.Package_Duration)
    parsed_PP_duration1 = parsed_PP_duration1.withColumn('Package_Duration',when(parsed_PP_duration1.Package_Duration_New2 < 7,'S').otherwise(parsed_PP_duration1.Package_Duration_New2)).drop(parsed_PP_duration1.Package_Duration_New2)
    parsed_PP_duration1 = parsed_PP_duration1.withColumn('Package_Duration_New2',when(parsed_PP_duration1.Package_Duration == 7,'M').otherwise(parsed_PP_duration1.Package_Duration)).drop(parsed_PP_duration1.Package_Duration)
    parsed_PP_duration1 = parsed_PP_duration1.withColumn('Package_Duration',when(parsed_PP_duration1.Package_Duration_New2 <= 15,'L').otherwise(parsed_PP_duration1.Package_Duration_New2)).drop(parsed_PP_duration1.Package_Duration_New2)
    parsed_PP_duration1 = parsed_PP_duration1.withColumn('Package_Duration_New2',when(parsed_PP_duration1.Package_Duration >= 30,'XL').otherwise(parsed_PP_duration1.Package_Duration)).drop(parsed_PP_duration1.Package_Duration)
    parsed_duration_t = parsed_PP_duration1.select(col('Package_Name'),col("Package_Duration_New2").alias("Package_Duration"),col('Price'),col('Package_Size'))
    parsed_duration_t2 = parsed_duration_t.withColumn('Price_New2',when(parsed_duration_t.Price <= 6.44,'XS').otherwise(parsed_duration_t.Price)).drop(parsed_duration_t.Price)
    parsed_duration_t2 = parsed_duration_t2.withColumn('Price',when(parsed_duration_t2.Price_New2 <= 27.62,'S').otherwise(parsed_duration_t2.Price_New2)).drop(parsed_duration_t2.Price_New2)
    parsed_duration_t2 = parsed_duration_t2.withColumn('Price_New2',when(parsed_duration_t2.Price <= 36,'M').otherwise(parsed_duration_t2.Price)).drop(parsed_duration_t2.Price)
    parsed_duration_t2 = parsed_duration_t2.withColumn('Price',when(parsed_duration_t2.Price_New2 <= 111,'L').otherwise(parsed_duration_t2.Price_New2)).drop(parsed_duration_t2.Price_New2)
    parsed_duration_t2 = parsed_duration_t2.withColumn('Price_New2',when(parsed_duration_t2.Price > 111,'XL').otherwise(parsed_duration_t2.Price)).drop(parsed_duration_t2.Price)
    parsed_master = parsed_duration_t2.select(col('Package_Name'),col("Package_Duration"),col('Price_New2').alias('Price'),col('Package_Size'))
    parsed_master_final = onehotenc(parsed_master, "Package_Duration")
    parsed_master_final = onehotenc(parsed_master_final, "Price")
    parsed_master_final = parsed_master_final.na.drop(subset=["Package_Size"])
    parsed_master_final = onehotenc(parsed_master_final, "Package_Size")
    parsed_master_final = parsed_master_final.join(master_mapping.select(col("Package Name").alias("Package_Name"),col("New_Package_Name")), ["Package_Name"], "left_outer")   
    parsed_master_final=parsed_master_final.withColumn("Package_Duration_XS", parsed_master_final["Package_Duration_XS"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Duration_S", parsed_master_final["Package_Duration_S"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Duration_M", parsed_master_final["Package_Duration_M"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Duration_L", parsed_master_final["Package_Duration_L"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Duration_XL", parsed_master_final["Package_Duration_XL"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Price_XS", parsed_master_final["Price_XS"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Price_S", parsed_master_final["Price_S"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Price_M", parsed_master_final["Price_M"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Price_L", parsed_master_final["Price_L"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Price_XL", parsed_master_final["Price_XL"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_1Mbps", parsed_master_final["Package_Size_1Mbps"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_256Kbps", parsed_master_final["Package_Size_256Kbps"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_2Mbps", parsed_master_final["Package_Size_2Mbps"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_384Kbps", parsed_master_final["Package_Size_384Kbps"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_4Mbps", parsed_master_final["Package_Size_4Mbps"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_512Kbps", parsed_master_final["Package_Size_512Kbps"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_64Kbps", parsed_master_final["Package_Size_64Kbps"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_6Mbps", parsed_master_final["Package_Size_6Mbps"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_Entertain", parsed_master_final["Package_Size_Entertain"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_Full speed", parsed_master_final["Package_Size_Full speed"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_Game", parsed_master_final["Package_Size_Game"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_Social", parsed_master_final["Package_Size_Social"].cast(IntegerType()))
    parsed_master_final=parsed_master_final.withColumn("Package_Size_Time", parsed_master_final["Package_Size_Time"].cast(IntegerType()))
    joinedtarif = df_ontop1802.join(master_jean3.select(col("PromotionCode").alias("package_id"),col("PackageName").alias("Unique_Name")), ["package_id"], "left_outer")
    customer_trans = joinedtarif.na.drop(subset=["Unique_Name"])
    customer_trans_simp = customer_trans.select(col("analytic_id"),col("total_transaction"),col("Unique_Name").alias("Package_Name")).na.drop(subset=["Package_Name"])
    customer_trans_final = onehotenc(customer_trans_simp, "Package_Name")
    customer_trans_final = multiply_all(customer_trans_final)
    customer_trans_final = customer_trans_final.drop('total_transaction')
    customer_trans_final.registerTempTable("cust_trans_final")
    #get customer transaction ontop list to filter out dimension
    column_seq = customer_trans_simp.select('Package_Name').distinct().rdd.map(lambda r: r[0]).collect()
    query = ''
    for c in column_seq:
        query =  query + ", SUM(Package_Name_" + c + ") AS " + c
    final = sqlContext.sql("SELECT analytic_id" + query + " FROM cust_trans_final GROUP BY analytic_id ")
    final = final.drop('7B20SMS10MMS1D')
    final.registerTempTable("final")
    
    #filter package according to existing in customer transaction and create index list of packages
    df_index = parsed_master_final.where(parsed_master_final.New_Package_Name.isin(column_seq)).drop_duplicates(subset=['New_Package_Name']).withColumn("id", monotonically_increasing_id())
    index_pp = df_index.select("id","Package_Duration_XS","Package_Duration_S","Package_Duration_M","Package_Duration_L","Package_Duration_XL","Price_XS","Price_S",\
               "Price_M","Price_L","Price_XL","Package_Size_1Mbps","Package_Size_256Kbps","Package_Size_2Mbps","Package_Size_384Kbps",\
               "Package_Size_4Mbps","Package_Size_512Kbps","Package_Size_64Kbps","Package_Size_6Mbps","Package_Size_Entertain",\
               "Package_Size_Full speed","Package_Size_Game","Package_Size_Social","Package_Size_Time")
	
    windowSpec = W.orderBy("id")
    ontop_rowid = df_index.withColumn("id", row_number().over(windowSpec))
    #df_index.select("New_Package_Name","id").withColumn("id", row_number().over(windowSpec)).show()
    ontop_rowid2 = ontop_rowid.withColumnRenamed("Package_Size_Full speed", "Package_Size_Full_speed")
    ontop_rowid2.registerTempTable("ontop_rowid2")
    index_pp2 = sqlContext.sql("SELECT id-1 as id,Package_Duration_XS,Package_Duration_S,Package_Duration_M,Package_Duration_L,Package_Duration_XL,Price_XS,Price_S,Price_M,Price_L,Price_XL,Package_Size_1Mbps,Package_Size_256Kbps,Package_Size_2Mbps,Package_Size_384Kbps,Package_Size_4Mbps,Package_Size_512Kbps,Package_Size_64Kbps,Package_Size_6Mbps,Package_Size_Entertain,Package_Size_Full_speed,Package_Size_Game,Package_Size_Social,Package_Size_Time FROM ontop_rowid2")
    package_sort = ontop_rowid.select('New_Package_Name').rdd.map(lambda r: r[0]).collect()
    query = ''
    for c in package_sort:
        query =  query + ", " + c
    final_selected = sqlContext.sql("SELECT analytic_id" + query + " FROM final")
    final_index = final_selected.withColumn("id", monotonically_increasing_id())
    windowSpec = W.orderBy("id")
    final_rowindex = final_index.withColumn("id", row_number().over(windowSpec))
    final_index.select("analytic_id","id").withColumn("id", row_number().over(windowSpec)).show()
    final_rowindex.registerTempTable("final_rowindex")
    index_ct = sqlContext.sql("SELECT id-1 as id" + query + " FROM final_rowindex")
    index_ct.registerTempTable("index_ct ")
    batch = sqlContext.sql("SELECT * FROM index_ct WHERE id BETWEEN 1 AND 20000")
    #matA = IndexedRowMatrix(index_ct.rdd.map(lambda row: IndexedRow(row[0], Vectors.dense(row[1:]))))
    matA = IndexedRowMatrix(batch.rdd.map(lambda row: IndexedRow(row[0], Vectors.dense(row[1:]))))
    bmA = matA.toBlockMatrix(colsPerBlock=222)
    ontop_pref_duration = index_pp2.select("id","Package_Duration_XS","Package_Duration_S","Package_Duration_M","Package_Duration_L","Package_Duration_XL")
    bmB_1 = IndexedRowMatrix(ontop_pref_duration.rdd.map(lambda x: IndexedRow(x[0], Vectors.dense(x[1:])))).toBlockMatrix(rowsPerBlock=222)
    del ontop_pref_duration
    del matA
    del index_ct
    del index_pp2
    del index_pp
    del final_rowindex
    del final_selected
    del parsed_master_final
    del customer_trans_final
    del customer_trans_simp
    del df_ontop1802
    del ontop_rowid2
    del ontop_rowid
    bm_duration_score_result = bmA.multiply(bmB_1)
    bm_duration_score_result.toIndexedRowMatrix().rows.sortBy(lambda x : x.index).map(lambda x:  Vectors.dense(x.vector)).saveAsTextFile("/preprocessed_cvm/duration_res_test")
    sc.stop()


