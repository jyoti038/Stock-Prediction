
import os
import sys

python_path = sys.executable
os.environ["PYSPARK_PYTHON"] = python_path
os.environ["PYSPARK_DRIVER_PYTHON"] = python_path

from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, max, min, count

spark = (
    SparkSession.builder
    .appName("StockMarketBigData")
    .master("local[1]")
    .config("spark.pyspark.python", python_path)
    .config("spark.pyspark.driver.python", python_path)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\n===== STOCK MARKET BIG DATA ANALYSIS =====")

file_path = os.path.abspath("data/prepared_stock_data.csv")

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(file_path)
)

print("\n1. Sample Stock Records")
df.show(5, truncate=False)

print("\n2. Dataset Schema")
df.printSchema()

print("\n3. Total Records")
print(df.count())

print("\n4. Stock Price Statistics")
df.select(
    count("Close").alias("Total_Records"),
    avg("Close").alias("Average_Close"),
    max("Close").alias("Maximum_Close"),
    min("Close").alias("Minimum_Close")
).show()

print("\n5. Prediction Target Distribution")
df.groupBy("Target").count().orderBy("Target").show()

print("\nSUCCESS: Real stock data analyzed using PySpark!")

spark.stop()
