# Microsoft Fabric notebook (PySpark). Attach a Lakehouse, upload these files to Files/raw/:
#   maude_events.parquet, recall.jsonl.gz, 510k.jsonl.gz, alerts.csv   (from this repo's pipeline)
# Produces Delta tables: bronze_maude, silver_maude_events, silver_maude_problems, silver_recalls,
#                        gold_monthly_firm_mode, gold_alerts
# and the gold tables are queryable from the Lakehouse SQL analytics endpoint.
from pyspark.sql import functions as F

RAW = "Files/raw"
CAT = {
    "leak": ["Leak/Splash", "Fluid/Blood Leak"],
    "contamination": ["Device Contamination with Chemical or Other Material", "Contamination",
                      "Contamination /Decontamination Problem", "Delivered as Unsterile Product"],
    "breakage_material": ["Break", "Crack", "Material Integrity Problem", "Material Separation", "Material Deformation",
                          "Material Puncture/Hole", "Material Twisted/Bent", "Detachment of Device or Device Component",
                          "Defective Component", "Material Too Rigid or Stiff"],
    "dosing_delivery": ["Volume Accuracy Problem", "Failure to Deliver", "Inaccurate Delivery",
                        "Insufficient Flow or Under Infusion", "Complete Blockage"],
    "mechanical_actuation": ["Physical Resistance/Sticking", "Mechanical Jam", "Difficult or Delayed Activation",
                             "Mechanical Problem", "Failure to Align", "Unintended Ejection", "Connection Problem",
                             "Loose or Intermittent Connection"],
    "labeling_packaging": ["Device Markings/Labelling Problem", "Packaging Problem", "Manufacturing, Packaging or Shipping Problem"],
    "software_electronics": ["Computer Software Problem", "Data Problem", "Wireless Communication Problem", "Battery Problem"],
    "use_interaction": ["Patient Device Interaction Problem", "Off-Label Use", "Improper or Incorrect Procedure or Method"],
}
term_map = spark.createDataFrame([(t, c) for c, ts in CAT.items() for t in ts], "term string, failure_mode string")

# ---- bronze: raw copy
bronze = spark.read.parquet(f"{RAW}/maude_events.parquet")
bronze.write.mode("overwrite").format("delta").saveAsTable("bronze_maude")

# ---- silver: typed events, one row per report
ev = (spark.table("bronze_maude")
      .withColumn("received", F.to_date("date_received", "yyyyMMdd"))
      .withColumn("event_date", F.to_date("date_of_event", "yyyyMMdd"))
      .withColumn("serious", F.col("event_type").isin("Injury", "Death"))
      .dropDuplicates(["mdr_report_key"]))
ev.drop("product_problems", "patient_problems").write.mode("overwrite").format("delta").saveAsTable("silver_maude_events")

probs = (ev.select("mdr_report_key", "received", "manufacturer", "product_code", F.explode_outer(F.array_distinct("product_problems")).alias("term"))
         .join(term_map, "term", "left"))
probs.write.mode("overwrite").format("delta").saveAsTable("silver_maude_problems")

rec = (spark.read.json(f"{RAW}/recall.jsonl.gz")
       .select("product_res_number", "res_event_number", F.to_date("event_date_initiated").alias("initiated"),
               "recalling_firm", "product_code", "recall_status", "root_cause_description", "reason_for_recall"))
rec.write.mode("overwrite").format("delta").saveAsTable("silver_recalls")

# ---- gold: monthly counts per manufacturer x failure mode, and detector alerts
gold = (spark.table("silver_maude_problems").where("failure_mode IS NOT NULL")
        .groupBy(F.date_trunc("month", "received").alias("month"), "manufacturer", "failure_mode")
        .agg(F.countDistinct("mdr_report_key").alias("reports")))
gold.write.mode("overwrite").format("delta").saveAsTable("gold_monthly_firm_mode")
spark.read.option("header", True).csv(f"{RAW}/alerts.csv").write.mode("overwrite").format("delta").saveAsTable("gold_alerts")

# ---- checks (fail loudly if the load is wrong)
n_raw, n_ev = bronze.count(), spark.table("silver_maude_events").count()
assert n_ev == bronze.select("mdr_report_key").distinct().count(), "silver lost or duplicated reports"
unmapped = spark.table("silver_maude_problems").where("term IS NOT NULL AND failure_mode IS NULL").count()
print({"bronze_rows": n_raw, "silver_events": n_ev, "problem_rows_unmapped_terms": unmapped,
       "gold_rows": spark.table("gold_monthly_firm_mode").count(), "recalls": spark.table("silver_recalls").count()})
