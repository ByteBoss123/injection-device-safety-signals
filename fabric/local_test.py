"""Run fabric/maude_lakehouse.py outside Fabric with local Spark + Delta Lake (same code path).
Usage: python fabric/local_test.py <dir containing Files/raw/...>"""
import os, sys, runpy
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

os.chdir(sys.argv[1])
b = (SparkSession.builder.appName("fabric-local").master("local[4]")
     .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
     .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
     .config("spark.driver.memory", "6g"))
spark = configure_spark_with_delta_pip(b).getOrCreate()
here = os.path.dirname(os.path.abspath(__file__))
runpy.run_path(os.path.join(here, "maude_lakehouse.py"), init_globals={"spark": spark})
for t in ["silver_maude_events", "gold_monthly_firm_mode", "gold_alerts", "silver_recalls"]:
    print(t, spark.table(t).count())
spark.sql("SELECT failure_mode, SUM(reports) n FROM gold_monthly_firm_mode GROUP BY 1 ORDER BY n DESC").show()
