# Databricks notebook source
# MAGIC %md
# MAGIC # อะไรทำให้แอปบน App Store "ติดตลาด"
# MAGIC วิเคราะห์ข้อมูล Apple App Store 1.23 ล้านแอป ด้วย PySpark, SQL และ scikit-learn บน Databricks
# MAGIC
# MAGIC **ข้อมูล:** [Kaggle: gauthamp10/apple-appstore-apps](https://www.kaggle.com/datasets/gauthamp10/apple-appstore-apps) (21 คอลัมน์, ไฟล์ลงวันที่ 14 ต.ค. 2021 ดูหมายเหตุเรื่องวันอ้างอิงในข้อ 12)
# MAGIC
# MAGIC **คำถามหลัก:** แอปแบบไหนได้รับความนิยม (วัดจากจำนวนรีวิว) และตลาดนี้เหลื่อมล้ำแค่ไหน
# MAGIC
# MAGIC ## นิยามที่ใช้ทั้ง notebook
# MAGIC | คำ | ความหมาย |
# MAGIC |---|---|
# MAGIC | ติดตลาด (hit) | แอปที่มีรีวิวมากกว่า 1,000 รีวิว |
# MAGIC | แอปร้าง (zombie) | แอปที่ไม่ได้อัปเดตเกิน 730 วัน นับถึงวันที่เก็บข้อมูล |
# MAGIC | วันอ้างอิง | วันที่ `Updated` มากที่สุดในข้อมูล (ไม่ใช่วันนี้ เพราะข้อมูลเป็น snapshot) |
# MAGIC | ช่วงปีที่ใช้ | 2008-2021 (ปี 2022 ข้อมูลไม่ครบปี และปี 2021 ก็มีข้อมูลถึงราว ต.ค. เท่านั้น) |
# MAGIC
# MAGIC ## สารบัญ
# MAGIC 0. เตรียมข้อมูลและสร้างตัวแปรใหม่
# MAGIC 1. ตรวจคุณภาพข้อมูล
# MAGIC 2. ตลาดเหลื่อมล้ำแค่ไหน
# MAGIC 3. หมวดหมู่: ติดตลาดง่ายและแอปร้าง
# MAGIC 4. ตลาดเปลี่ยนไปตามเวลา
# MAGIC 5. ราคาและแอปฟรี
# MAGIC 6. ขนาดแอป
# MAGIC 7. นักพัฒนา: รายเดี่ยวและโรงงานแอป
# MAGIC 8. การอัปเดตกับความนิยม
# MAGIC 9. โมเดลทำนายแอปที่ติดตลาด
# MAGIC 10. ทดสอบสมมติฐานทางสถิติ
# MAGIC    - 10.1 Correlation
# MAGIC    - 10.2 โมเดลบน Spark MLlib (LR / RF / GBT + baseline)
# MAGIC    - 10.3 จัดกลุ่มแอปด้วย k-means
# MAGIC    - 10.4 สรุปและคำแนะนำ
# MAGIC 11. ตารางสรุปสำหรับทำเว็บ
# MAGIC 12. ข้อจำกัดของการวิเคราะห์

# COMMAND ----------
# MAGIC %md
# MAGIC ## คู่มือสำหรับทำสไลด์
# MAGIC ตารางนี้บอกว่าสไลด์แต่ละหน้า (นำเสนอ 15 นาที 14 สไลด์) ใช้ผลจากหัวข้อไหนใน notebook
# MAGIC
# MAGIC | สไลด์ | เรื่อง | ดูหัวข้อ | ไฟล์ใน `results/` |
# MAGIC |---|---|---|---|
# MAGIC | 1 | ชื่อเรื่องและคำถามหลัก | หัว notebook | – |
# MAGIC | 2 | Dataset & เครื่องมือ | หัว notebook, 0 | – |
# MAGIC | 3 | Pipeline, นิยาม hit / zombie, สัดส่วน hit | 0, ตารางนิยาม, 10.2 | – |
# MAGIC | 4 | คุณภาพข้อมูล และข้อมูลผิดตรงไหน | 1, 12 | `quality.csv` |
# MAGIC | 5 | ตลาดเหลื่อมล้ำ | 2 | `Long Tail Review.csv`, `review.csv` |
# MAGIC | 6 | หมวดหมู่ & เวลา | 3, 4 | `best category.csv`, `zombieapp.csv`, `ratio per year.csv`, `timeline open app free.paid.csv` |
# MAGIC | 7 | ราคา & ขนาด | 5, 6 | `free.paid.csv`, `price psycho.csv`, `fatapps.csv`, `fatapp catagory.csv` |
# MAGIC | 8 | นักพัฒนา | 7 | `factory vs indie.csv`, `top 15 dev most app.csv` |
# MAGIC | 9 | การอัปเดต | 8 | `is update effect review.csv`, `good app vs bad app.csv` |
# MAGIC | 10 | สถิติ & correlation | 10, 10.1 | – |
# MAGIC | 11 | โมเดลทำนาย hit | 9, 10.2 | – |
# MAGIC | 12 | กลุ่มแอป (k-means) | 10.3 | – |
# MAGIC | 13 | Demo เว็บ | 11 | `cube for web.csv` |
# MAGIC | 14 | สรุป คำแนะนำ และข้อจำกัด | 10.4, 12 | – |

# COMMAND ----------
# MAGIC %md
# MAGIC ## 0. เตรียมข้อมูลและสร้างตัวแปรใหม่
# MAGIC **ทำอะไร:** อ่านตารางที่อัปโหลดไว้ ตัดแถวซ้ำตาม `App_Id` แล้วสร้างคอลัมน์ที่ใช้วิเคราะห์ต่อ
# MAGIC
# MAGIC | ตัวแปรใหม่ | วิธีคำนวณ |
# MAGIC |---|---|
# MAGIC | `size_mb` | `Size_Bytes` หาร 1,000,000 |
# MAGIC | `release_year` | ปีของ `Released` |
# MAGIC | `age_days` | วันอ้างอิง ลบ `Released` |
# MAGIC | `days_since_update` | วันอ้างอิง ลบ `Updated` |
# MAGIC | `is_zombie` | 1 ถ้า `days_since_update` มากกว่า 730 |
# MAGIC | `has_website` | 1 ถ้า `Developer_Website` ไม่ใช่ null |
# MAGIC | `ios_ver` | เลขเวอร์ชันหลักของ iOS ที่ต้องใช้ เช่น `5.1.1` เป็น 5 |
# MAGIC | `dev_app_count` | จำนวนแอปทั้งหมดของนักพัฒนาคนเดียวกัน |
# MAGIC
# MAGIC ผลลัพธ์ถูกเก็บเป็น view ชื่อ `apps` ที่เซลล์ SQL ทุกอันใช้ต่อ

# COMMAND ----------
from pyspark.sql import functions as F
from pyspark.sql.window import Window

T = "workspace.default.apple_app_data"
raw = spark.table(T)

ref_date = raw.agg(F.max("Updated")).first()[0]
print("ref_date:", ref_date)

clean = (
    raw.dropDuplicates(["App_Id"])
    .withColumn("size_mb", F.col("Size_Bytes") / 1e6)
    .withColumn("release_year", F.year("Released"))
    .withColumn("age_days", F.datediff(F.lit(ref_date), F.col("Released")))
    .withColumn("days_since_update", F.datediff(F.lit(ref_date), F.col("Updated")))
    .withColumn("is_zombie", (F.col("days_since_update") > 730).cast("int"))
    .withColumn("has_website", F.col("Developer_Website").isNotNull().cast("int"))
    .withColumn("ios_ver", F.regexp_extract("Required_IOS_Version", r"^(\d+)", 1).cast("int"))
    .withColumn("dev_app_count", F.count("*").over(Window.partitionBy("DeveloperId")))
)
clean.createOrReplaceTempView("apps")
print("rows:", clean.count())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. ตรวจคุณภาพข้อมูล
# MAGIC **ทำอะไร:** นับค่า null ทุกคอลัมน์ นับแอปที่ไม่มีรีวิว และตรวจว่าคะแนนเวอร์ชันปัจจุบันต่างจากคะแนนรวมหรือไม่
# MAGIC
# MAGIC **วัดยังไง:** นับจำนวนแถวที่เข้าเงื่อนไข แล้วเทียบเป็นเปอร์เซ็นต์ของทั้งหมด
# MAGIC
# MAGIC **ที่พบ:**
# MAGIC - null จริงมีมากแค่ `Developer_Website` (ราว 52%) ซึ่งหมายถึงนักพัฒนาไม่มีเว็บไซต์ จึงใช้เป็นตัวแปรได้
# MAGIC - แอป 55.6% ไม่มีรีวิวเลย (เก็บเป็น 0 ไม่ใช่ null) ดังนั้นการวิเคราะห์ดาวต้องกรอง `Reviews > 0`
# MAGIC - `Current_Version_Score` ไม่ต่างจาก `Average_User_Rating` ในแอปที่มีรีวิวเวอร์ชันปัจจุบันตั้งแต่ 50 ขึ้นไป จึงสร้างตัวแปร "แนวโน้มคะแนน" ไม่ได้

# COMMAND ----------
display(raw.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in raw.columns]))

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT COUNT(*) AS total,
# MAGIC        SUM(CASE WHEN Reviews = 0 THEN 1 ELSE 0 END) AS no_reviews,
# MAGIC        ROUND(100.0 * SUM(CASE WHEN Reviews = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_no_reviews
# MAGIC FROM apps;

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT COUNT(*) AS total,
# MAGIC        SUM(CASE WHEN Current_Version_Score <> Average_User_Rating THEN 1 ELSE 0 END) AS differ
# MAGIC FROM apps
# MAGIC WHERE Current_Version_Reviews >= 50;

# COMMAND ----------
print("apps updated within 90 days of ref_date:", clean.where("days_since_update <= 90").count())

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. ตลาดเหลื่อมล้ำแค่ไหน
# MAGIC **ทำอะไร:** ดูว่ารีวิวทั้งหมดในตลาดกระจุกอยู่กับแอปกลุ่มไหน
# MAGIC
# MAGIC **วัดยังไง:**
# MAGIC 1. เรียงแอปจากรีวิวมากไปน้อย แล้วคำนวณสัดส่วนรีวิวที่แอป 1% และ 10% บนสุดถือ
# MAGIC 2. วาด **Lorenz curve** แกนนอนคือสัดส่วนแอป (เรียงจากรีวิวน้อยไปมาก) แกนตั้งคือสัดส่วนรีวิวสะสม เส้นประคือกรณีที่ทุกแอปได้รีวิวเท่ากัน
# MAGIC 3. คำนวณ **Gini** จากพื้นที่ใต้เส้น Lorenz (0 = เท่ากันหมด, 1 = แอปเดียวถือทุกอย่าง)
# MAGIC
# MAGIC **อ่านผล:** เส้นยิ่งโค้งห่างจากเส้นประ Gini ยิ่งสูง ตลาดยิ่งกระจุกตัว
# MAGIC
# MAGIC **ข้อควรระวัง:** วัดจากจำนวนรีวิว ไม่ใช่จำนวนผู้ใช้หรือยอดดาวน์โหลด แอปที่ไม่มีรีวิวไม่ได้แปลว่าไม่มีผู้ใช้

# COMMAND ----------
# MAGIC %sql
# MAGIC WITH r AS (
# MAGIC   SELECT Reviews, PERCENT_RANK() OVER (ORDER BY Reviews DESC) AS pct FROM apps
# MAGIC )
# MAGIC SELECT ROUND(SUM(CASE WHEN pct <= 0.01 THEN Reviews END) / SUM(Reviews) * 100, 1) AS top1pct_share,
# MAGIC        ROUND(SUM(CASE WHEN pct <= 0.10 THEN Reviews END) / SUM(Reviews) * 100, 1) AS top10pct_share
# MAGIC FROM r;

# COMMAND ----------
import numpy as np
import matplotlib.pyplot as plt

rev = np.sort(clean.select("Reviews").toPandas()["Reviews"].fillna(0).values)
cum = np.cumsum(rev) / rev.sum()
x = np.arange(1, len(rev) + 1) / len(rev)
gini = 1 - 2 * np.trapezoid(cum, x)

plt.figure(figsize=(6, 6))
plt.plot(x, cum, label=f"Lorenz (Gini = {gini:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect equality")
plt.xlabel("Share of apps (low to high reviews)")
plt.ylabel("Share of total reviews")
plt.title("Review inequality across the App Store")
plt.legend()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. หมวดหมู่: ติดตลาดง่ายและแอปร้าง
# MAGIC **ทำอะไร:** เปรียบเทียบแต่ละ `Primary_Genre` ด้วยสามตัวชี้วัด
# MAGIC
# MAGIC | ตัวชี้วัด | วิธีคำนวณ |
# MAGIC |---|---|
# MAGIC | `pct_any_review` | แอปที่มีรีวิวอย่างน้อย 1 รายการ หารจำนวนแอปในหมวด |
# MAGIC | `pct_hit` | แอปที่มีรีวิวเกิน 1,000 หารจำนวนแอปในหมวด |
# MAGIC | `pct_zombie` | แอปที่ไม่อัปเดตเกิน 2 ปี หารจำนวนแอปในหมวด |
# MAGIC
# MAGIC **อ่านผล:** หมวดที่ `pct_hit` สูงและ `pct_zombie` ต่ำคือหมวดที่แอปมีโอกาสอยู่รอดและดังมากกว่า ให้ดูคอลัมน์ `apps` ควบคู่เสมอ เพราะหมวดเล็กมากทำให้เปอร์เซ็นต์แกว่งง่าย

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT Primary_Genre, COUNT(*) AS apps,
# MAGIC        ROUND(100.0 * AVG(CASE WHEN Reviews > 0 THEN 1 ELSE 0 END), 1) AS pct_any_review,
# MAGIC        ROUND(100.0 * AVG(CASE WHEN Reviews > 1000 THEN 1 ELSE 0 END), 2) AS pct_hit
# MAGIC FROM apps
# MAGIC WHERE release_year <= 2021
# MAGIC GROUP BY Primary_Genre
# MAGIC ORDER BY pct_hit DESC;

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT Primary_Genre, COUNT(*) AS apps,
# MAGIC        ROUND(100.0 * AVG(is_zombie), 1) AS pct_zombie
# MAGIC FROM apps
# MAGIC GROUP BY Primary_Genre
# MAGIC HAVING COUNT(*) >= 1000
# MAGIC ORDER BY pct_zombie DESC;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. ตลาดเปลี่ยนไปตามเวลา
# MAGIC **ทำอะไร:** ดูจำนวนแอปที่เปิดตัวแต่ละปี แยกแอปฟรีและเสียเงิน และดูสัดส่วนแอปที่ติดตลาดของแต่ละรุ่นปี
# MAGIC
# MAGIC **วัดยังไง:** นับแอปตาม `release_year` และ `Free` และคำนวณ `pct_hit` ของแต่ละรุ่นปี
# MAGIC
# MAGIC **ข้อควรระวัง:** แอปรุ่นใหม่มีเวลาสะสมรีวิวน้อยกว่าโดยธรรมชาติ ตัวเลขของปี 2020-2021 จึงต่ำกว่ารุ่นเก่าส่วนหนึ่งเพราะเหตุนี้ ไม่ได้แปลว่าคุณภาพแอปลดลงทั้งหมด

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT release_year, Free, COUNT(*) AS apps
# MAGIC FROM apps
# MAGIC WHERE release_year BETWEEN 2008 AND 2021
# MAGIC GROUP BY release_year, Free
# MAGIC ORDER BY release_year;

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT release_year, COUNT(*) AS apps,
# MAGIC        ROUND(100.0 * AVG(CASE WHEN Reviews > 0 THEN 1 ELSE 0 END), 1) AS pct_any_review,
# MAGIC        ROUND(100.0 * AVG(CASE WHEN Reviews > 1000 THEN 1 ELSE 0 END), 2) AS pct_hit
# MAGIC FROM apps
# MAGIC WHERE release_year BETWEEN 2008 AND 2021
# MAGIC GROUP BY release_year
# MAGIC ORDER BY release_year;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. ราคาและแอปฟรี
# MAGIC **ทำอะไร:** ดูว่านักพัฒนาตั้งราคาที่จุดไหนมากที่สุด และแอปฟรีต่างจากแอปเสียเงินอย่างไร
# MAGIC
# MAGIC **วัดยังไง:**
# MAGIC - นับจำนวนแอปในแต่ละราคา (เฉพาะสกุล USD, ราคา 0-20 ดอลลาร์)
# MAGIC - เปรียบเทียบจำนวนรีวิวและดาวเฉลี่ยของแอปฟรีกับเสียเงิน เฉพาะแอปที่มีรีวิว โดยรายงานทั้งค่าเฉลี่ยและค่ามัธยฐาน
# MAGIC
# MAGIC **อ่านผล:** ค่าเฉลี่ยรีวิวถูกดึงขึ้นด้วยแอปดังไม่กี่ตัว ค่ามัธยฐานสะท้อนแอปทั่วไปมากกว่า จึงต้องดูคู่กัน

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT Price, COUNT(*) AS apps
# MAGIC FROM apps
# MAGIC WHERE Price > 0 AND Price <= 20 AND Currency = 'USD'
# MAGIC GROUP BY Price
# MAGIC ORDER BY apps DESC
# MAGIC LIMIT 20;

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT Free, COUNT(*) AS apps,
# MAGIC        ROUND(AVG(Reviews), 1) AS avg_reviews,
# MAGIC        percentile_approx(Reviews, 0.5) AS median_reviews,
# MAGIC        ROUND(AVG(Average_User_Rating), 3) AS avg_rating
# MAGIC FROM apps
# MAGIC WHERE Reviews > 0
# MAGIC GROUP BY Free;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. ขนาดแอป
# MAGIC **ทำอะไร:** ดูว่าแอปใหญ่ขึ้นตามปีหรือไม่ ขนาดต่างกันตามหมวดอย่างไร และขนาดสัมพันธ์กับการติดตลาดหรือไม่
# MAGIC
# MAGIC **วัดยังไง:**
# MAGIC - ค่าเฉลี่ยและมัธยฐานของ `size_mb` ตามปีและตามหมวด
# MAGIC - แบ่งขนาดเป็น 5 ช่วง (ต่ำกว่า 10, 10-50, 50-100, 100-500, 500 MB ขึ้นไป) แล้วคำนวณ `pct_hit` ของแต่ละช่วง
# MAGIC
# MAGIC **อ่านผล:** ใช้มัธยฐานเป็นหลัก เพราะแอปขนาดใหญ่ผิดปกติดึงค่าเฉลี่ยขึ้น ขนาดสัมพันธ์กับการติดตลาด แต่ไม่ได้พิสูจน์ว่าการทำแอปให้ใหญ่ขึ้นทำให้ดัง

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT release_year,
# MAGIC        ROUND(AVG(size_mb), 1) AS avg_size_mb,
# MAGIC        ROUND(percentile_approx(size_mb, 0.5), 1) AS median_size_mb
# MAGIC FROM apps
# MAGIC WHERE release_year BETWEEN 2008 AND 2021 AND size_mb > 0
# MAGIC GROUP BY release_year
# MAGIC ORDER BY release_year;

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT Primary_Genre, ROUND(percentile_approx(size_mb, 0.5), 1) AS median_size_mb
# MAGIC FROM apps
# MAGIC WHERE size_mb > 0
# MAGIC GROUP BY Primary_Genre
# MAGIC ORDER BY median_size_mb DESC;

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT CASE WHEN size_mb IS NULL THEN 'unknown'
# MAGIC             WHEN size_mb < 10  THEN '1) <10 MB'
# MAGIC             WHEN size_mb < 50  THEN '2) 10-50 MB'
# MAGIC             WHEN size_mb < 100 THEN '3) 50-100 MB'
# MAGIC             WHEN size_mb < 500 THEN '4) 100-500 MB'
# MAGIC             ELSE '5) 500+ MB' END AS size_bucket,
# MAGIC        COUNT(*) AS apps,
# MAGIC        ROUND(100.0 * AVG(CASE WHEN Reviews > 1000 THEN 1 ELSE 0 END), 2) AS pct_hit
# MAGIC FROM apps
# MAGIC WHERE release_year BETWEEN 2008 AND 2021
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. นักพัฒนา: รายเดี่ยวและโรงงานแอป
# MAGIC **ทำอะไร:** แบ่งนักพัฒนาตามจำนวนแอปที่ตัวเองมี แล้วเปรียบเทียบผลงานของแต่ละกลุ่ม
# MAGIC
# MAGIC **วัดยังไง:** แบ่งกลุ่มจาก `dev_app_count` เป็น 1 แอป, 2-10, 11-100 และมากกว่า 100 แอป จากนั้นดูจำนวนรีวิวเฉลี่ย ค่ามัธยฐาน และดาวเฉลี่ย (เฉพาะแอปที่มีรีวิว) พร้อมรายชื่อนักพัฒนา 15 อันดับที่มีแอปมากสุด
# MAGIC
# MAGIC **อ่านผล:** นักพัฒนารายใหญ่หลายรายมีรีวิวเฉลี่ยใกล้ 0 ซึ่งบ่งว่าอาจเป็นผู้ผลิตแอปแบรนด์ให้ลูกค้าจำนวนมาก (ข้อสันนิษฐาน ยังไม่ได้ตรวจชื่อแอปทีละตัว)

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT CASE WHEN dev_app_count = 1 THEN '1 app (indie)'
# MAGIC             WHEN dev_app_count <= 10 THEN '2-10 apps'
# MAGIC             WHEN dev_app_count <= 100 THEN '11-100 apps'
# MAGIC             ELSE '100+ apps (factory)' END AS dev_size,
# MAGIC        COUNT(*) AS apps,
# MAGIC        ROUND(AVG(Reviews), 1) AS avg_reviews,
# MAGIC        percentile_approx(Reviews, 0.5) AS median_reviews,
# MAGIC        ROUND(AVG(CASE WHEN Reviews > 0 THEN Average_User_Rating END), 3) AS avg_rating
# MAGIC FROM apps
# MAGIC GROUP BY 1
# MAGIC ORDER BY apps;

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT Developer, COUNT(*) AS apps, ROUND(AVG(Reviews), 1) AS avg_reviews
# MAGIC FROM apps
# MAGIC GROUP BY Developer
# MAGIC ORDER BY apps DESC
# MAGIC LIMIT 15;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. การอัปเดตกับความนิยม
# MAGIC **ทำอะไร:** ดูว่าแอปที่อัปเดตล่าสุดไม่นานมีโอกาสได้รีวิวและติดตลาดมากกว่าหรือไม่
# MAGIC
# MAGIC **วัดยังไง:** แบ่งแอปตามช่วงเวลาตั้งแต่อัปเดตล่าสุด (ภายใน 3 เดือน, 3-12 เดือน, 1-2 ปี, เกิน 2 ปี) แล้วเทียบ `pct_any_review` และ `pct_hit` เพื่อไม่ให้อายุแอปมาปนผล เลือกเฉพาะแอปที่เปิดตัวปี 2016-2018
# MAGIC
# MAGIC **อ่านผล:** แอปที่อัปเดตไม่นานติดตลาดมากกว่าอย่างชัดเจน แต่แปลได้สองทาง คือการอัปเดตช่วยให้ดัง หรือแอปที่ดังมีรายได้จึงอัปเดตต่อ ข้อมูล snapshot นี้แยกสองทางไม่ได้ กลุ่ม "ภายใน 3 เดือน" ไม่มีแอปเลย ซึ่งอาจแปลว่าวันอ้างอิงมาจากแอปไม่กี่ตัว

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT CASE WHEN days_since_update <= 90  THEN '1) within 3 months'
# MAGIC             WHEN days_since_update <= 365 THEN '2) 3-12 months'
# MAGIC             WHEN days_since_update <= 730 THEN '3) 1-2 years'
# MAGIC             ELSE '4) over 2 years' END AS last_update,
# MAGIC        COUNT(*) AS apps,
# MAGIC        ROUND(100.0 * AVG(CASE WHEN Reviews > 0 THEN 1 ELSE 0 END), 1) AS pct_any_review,
# MAGIC        ROUND(100.0 * AVG(CASE WHEN Reviews > 1000 THEN 1 ELSE 0 END), 2) AS pct_hit
# MAGIC FROM apps
# MAGIC WHERE release_year BETWEEN 2016 AND 2018
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. โมเดลทำนายแอปที่ติดตลาด
# MAGIC **ทำอะไร:** สร้างโมเดลทำนายว่าแอปจะมีรีวิวเกิน 1,000 หรือไม่ จากลักษณะของแอป แล้วดูว่าปัจจัยไหนช่วยทำนายได้มากที่สุด
# MAGIC
# MAGIC | หัวข้อ | รายละเอียด |
# MAGIC |---|---|
# MAGIC | เป้าหมาย | `Reviews > 1000` (ติดตลาดหรือไม่) |
# MAGIC | ตัวแปรต้น | หมวด, ฟรี/เสียเงิน, ราคา, ขนาด, มีเว็บไซต์, iOS ขั้นต่ำ, อายุแอป, จำนวนแอปของนักพัฒนา |
# MAGIC | ข้อมูล | สุ่ม 25% ของแอป แบ่ง train 80% และ test 20% แบบรักษาสัดส่วนคลาส |
# MAGIC | โมเดล | Histogram Gradient Boosting (scikit-learn) |
# MAGIC | ตัววัดโมเดล | **AUC** เพราะแอปที่ติดตลาดมีแค่ราว 2% การวัดด้วย accuracy จะหลอกตา (ตอบ "ไม่ติด" ทุกแอปก็ได้ 98%) |
# MAGIC | ตัววัดความสำคัญ | **Permutation importance** สลับค่าของตัวแปรหนึ่งแล้วดูว่า AUC ตกเท่าไร ตกมากแปลว่าสำคัญมาก |
# MAGIC
# MAGIC **หมายเหตุ:** `age_days` ต้องอยู่ในโมเดล ไม่งั้นโมเดลจะเรียนรู้แค่ "แอปเก่ามีเวลาสะสมรีวิวนาน"
# MAGIC
# MAGIC **ข้อควรระวัง:** ความสำคัญของตัวแปรคือความสามารถในการ **ทำนาย** ไม่ใช่ **สาเหตุ** และตัวแปรที่สัมพันธ์กัน (เช่น อายุแอปกับ iOS ขั้นต่ำ) จะแบ่งความสำคัญกัน

# COMMAND ----------
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

cols = ["Primary_Genre", "Free", "Price", "size_mb", "has_website",
        "ios_ver", "age_days", "dev_app_count", "Reviews"]
pdf = (clean.select(*cols).where("age_days IS NOT NULL")
       .sample(fraction=0.25, seed=42).toPandas())

pdf["y"] = (pdf["Reviews"] > 1000).astype(int)
pdf["Free"] = pdf["Free"].astype(int)
pdf["Primary_Genre"] = pdf["Primary_Genre"].astype("category")
X = pdf.drop(columns=["Reviews", "y"])
y = pdf["y"]
print("positive rate:", round(y.mean() * 100, 2), "%  rows:", len(pdf))

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=42)
model.fit(Xtr, ytr)
print("AUC:", round(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]), 4))

# COMMAND ----------
imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=42)
imp_df = pd.Series(imp.importances_mean, index=X.columns).sort_values()
imp_df.plot.barh(figsize=(7, 4), title="Permutation importance (drop in AUC)")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. ทดสอบสมมติฐานทางสถิติ
# MAGIC **ทำอะไร:** เปรียบเทียบดาวเฉลี่ยของสามคู่กลุ่ม (เฉพาะแอปที่มีรีวิว สุ่ม 20%)
# MAGIC
# MAGIC | คู่เปรียบเทียบ | คำถาม |
# MAGIC |---|---|
# MAGIC | ฟรี vs เสียเงิน | แอปฟรีได้ดาวต่างจากแอปเสียเงินหรือไม่ |
# MAGIC | มีเว็บไซต์ vs ไม่มี | นักพัฒนาที่มีเว็บไซต์ได้ดาวต่างกันหรือไม่ |
# MAGIC | นักพัฒนารายใหญ่ (มากกว่า 10 แอป) vs รายเดี่ยว (1 แอป) | รายใหญ่ได้ดาวสูงกว่าหรือไม่ |
# MAGIC
# MAGIC **วัดยังไง:**
# MAGIC - **Mann-Whitney U test** เพราะดาวไม่ได้กระจายแบบปกติ ให้ค่า p-value
# MAGIC - **ผลต่างของดาวเฉลี่ย** บอกขนาดผลในหน่วยที่เข้าใจง่าย
# MAGIC - **Cliff's delta** ขนาดผล (effect size) ตั้งแต่ -1 ถึง 1 เกณฑ์คร่าว ๆ คือ ราว 0.15 เล็ก, 0.33 กลาง, 0.47 ใหญ่
# MAGIC
# MAGIC **อ่านผล:** ข้อมูลมีหลายแสนแถว p-value จึงเล็กมากเกือบทุกกรณี แม้ผลจริงเล็กจนไม่มีความหมาย ให้ตัดสินจาก Cliff's delta และผลต่างของดาวเป็นหลัก

# COMMAND ----------
from scipy.stats import mannwhitneyu

s = (clean.where("Reviews > 0").select("Average_User_Rating", "Free", "has_website", "dev_app_count")
     .sample(fraction=0.2, seed=1).toPandas())

def test(a, b, name):
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    cliff = 2 * u / (len(a) * len(b)) - 1
    print(f"{name}: p={p:.2e}, mean diff={a.mean() - b.mean():+.3f} stars, Cliff's delta={cliff:+.3f}")

test(s[s.Free == True].Average_User_Rating, s[s.Free == False].Average_User_Rating, "Free vs Paid (rating)")
test(s[s.has_website == 1].Average_User_Rating, s[s.has_website == 0].Average_User_Rating, "Has website vs none (rating)")
test(s[s.dev_app_count > 10].Average_User_Rating, s[s.dev_app_count == 1].Average_User_Rating, "Big dev vs indie (rating)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10.1 ความสัมพันธ์ระหว่างตัวแปร (Correlation)
# MAGIC **ทำอะไร:** ดูว่าตัวแปรตัวเลขแต่ละคู่เคลื่อนไปด้วยกันแค่ไหน ใช้ประกอบการอ่านโมเดลข้อ 10.2 เพราะตัวแปรที่สัมพันธ์กันสูงจะแบ่งความสำคัญกัน
# MAGIC
# MAGIC **วัดยังไง:** Spearman correlation ด้วย `pyspark.ml.stat.Correlation` บนข้อมูลทั้งชุด (ปี 2008–2021) Spearman ใช้อันดับแทนค่าจริง จึงไม่ถูกแอปรีวิวหลักล้านไม่กี่ตัวดึงผล
# MAGIC
# MAGIC **อ่านผลยังไง:** ค่าอยู่ระหว่าง -1 ถึง 1 ใกล้ 0 คือไม่สัมพันธ์ ราว ±0.3 คือปานกลาง เกิน ±0.5 คือสูง
# MAGIC
# MAGIC **ผลที่ได้**
# MAGIC - `Reviews` กับ `Average_User_Rating` ได้ 0.88 แต่**อย่าอ่านว่ารีวิวเยอะแล้วดาวสูง** แอปที่ไม่มีรีวิว (684,320 แอป) ถูกเก็บดาวเป็น 0 ทุกตัว ค่านี้จึงสะท้อนแค่ว่าแอปมีรีวิวหรือไม่
# MAGIC - ตัวแปรเวลาสัมพันธ์กันสูง: `ios_ver` กับ `days_since_update` -0.65, `age_days` กับ `days_since_update` 0.57, `ios_ver` กับ `age_days` -0.46 (แอปเก่ารองรับ iOS รุ่นต่ำและไม่ได้อัปเดตนาน) ตัวแปรกลุ่มนี้จึงแบ่งความสำคัญกันในโมเดลข้อ 10.2
# MAGIC - ตัวแปรอื่นแทบไม่สัมพันธ์กัน (|r| < 0.3) จึงใส่ในโมเดลพร้อมกันได้

# COMMAND ----------
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import numpy as np

corr_cols = ["Reviews", "Average_User_Rating", "Price", "size_mb", "ios_ver",
             "age_days", "days_since_update", "dev_app_count", "has_website"]
corr_df = (clean.where("release_year BETWEEN 2008 AND 2021")
           .select([F.col(c).cast("double") for c in corr_cols]).na.drop())
vec = VectorAssembler(inputCols=corr_cols, outputCol="v").transform(corr_df).select("v")
m = Correlation.corr(vec, "v", "spearman").head()[0].toArray()

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
ax.set_yticks(range(len(corr_cols)), corr_cols)
for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=8)
fig.colorbar(im)
ax.set_title("Spearman correlation (apps released 2008-2021)")
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10.2 โมเดลบน Spark MLlib: เทียบ 3 algorithm และ baseline
# MAGIC **ทำอะไร:** สร้างโมเดลทำนาย hit ใหม่ด้วย Spark MLlib บนข้อมูล**ทั้งชุด** (ไม่ต้อง `toPandas()`) เทียบ 3 algorithm แล้วตัดตัวแปรออกทีละชุด เพื่อดูว่าโมเดลเรียนรู้อะไรจริง
# MAGIC
# MAGIC | หัวข้อ | รายละเอียด |
# MAGIC |---|---|
# MAGIC | ข้อมูล | แอปที่เปิดตัวปี 2008–2021 ทั้งหมด แบ่ง train 80% / test 20% (`randomSplit`, seed 42) |
# MAGIC | ตัวแปรต้น | ชุดเดียวกับข้อ 9: หมวด, ฟรี/เสียเงิน, ราคา, ขนาด, มีเว็บไซต์, iOS ขั้นต่ำ, อายุแอป, จำนวนแอปของนักพัฒนา |
# MAGIC | Pipeline | `StringIndexer` → `OneHotEncoder` (หมวด) → `Imputer` (ค่าว่างใช้ median) → `VectorAssembler` → โมเดล |
# MAGIC | โมเดล | Logistic Regression, Random Forest, Gradient-Boosted Trees |
# MAGIC | คลาสไม่สมดุล | hit มีราว 2% จึงให้น้ำหนัก (`weightCol`) กับแอปที่เป็น hit มากขึ้น |
# MAGIC | ตัววัด | **ROC-AUC** และ **PR-AUC** (PR-AUC เหมาะกับคลาสไม่สมดุล ค่าฐานของการเดาสุ่มเท่ากับสัดส่วน hit ไม่ใช่ 0.5) |
# MAGIC
# MAGIC **Baseline:** ใช้ Random Forest กับชุดตัวแปร 5 แบบ ถ้าใช้แค่ `age_days` แล้ว AUC ใกล้กับโมเดลเต็ม แปลว่าโมเดลแค่รู้ว่าแอปเก่ามีเวลาสะสมรีวิวนาน
# MAGIC
# MAGIC **หมายเหตุ:** ถ้ารันช้าเกินไป ให้ลด `FRACTION` ในเซลล์ถัดไปเหลือ 0.3 แล้วเขียนไว้ในผล

# COMMAND ----------
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, Imputer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

FRACTION = 1.0
NUM = ["Price", "size_mb", "ios_ver", "age_days", "dev_app_count", "has_website", "free_int"]

ml_df = (clean.where("release_year BETWEEN 2008 AND 2021 AND age_days IS NOT NULL")
         .select("Primary_Genre", "Free", "Price", "size_mb", "has_website", "ios_ver",
                 "age_days", "dev_app_count", "Reviews")
         .withColumn("label", (F.col("Reviews") > 1000).cast("double"))
         .withColumn("free_int", F.col("Free").cast("double"))
         .withColumn("Primary_Genre", F.coalesce("Primary_Genre", F.lit("Unknown")))
         .withColumn("dev_app_count", F.col("dev_app_count").cast("double"))
         .withColumn("has_website", F.col("has_website").cast("double"))
         .withColumn("ios_ver", F.col("ios_ver").cast("double"))
         .withColumn("age_days", F.col("age_days").cast("double")))
if FRACTION < 1.0:
    ml_df = ml_df.sample(fraction=FRACTION, seed=42)

pos_rate = ml_df.agg(F.avg("label")).first()[0]
ml_df = ml_df.withColumn("weight", F.when(F.col("label") == 1, 0.5 / pos_rate).otherwise(0.5 / (1 - pos_rate)))
train, test = ml_df.randomSplit([0.8, 0.2], seed=42)
print(f"rows: {ml_df.count():,}  hit rate (class balance): {pos_rate * 100:.2f}%  (PR-AUC baseline = {pos_rate:.4f})")

roc_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
pr_eval = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")

def build(model, num_cols, use_genre=True):
    stages = [Imputer(inputCols=num_cols, outputCols=[c + "_imp" for c in num_cols], strategy="median")]
    inputs = [c + "_imp" for c in num_cols]
    if use_genre:
        stages += [StringIndexer(inputCol="Primary_Genre", outputCol="genre_idx", handleInvalid="keep"),
                   OneHotEncoder(inputCol="genre_idx", outputCol="genre_vec")]
        inputs.append("genre_vec")
    stages += [VectorAssembler(inputCols=inputs, outputCol="features"), model]
    return Pipeline(stages=stages)

def evaluate(pipe):
    fitted = pipe.fit(train)
    pred = fitted.transform(test)
    return fitted, roc_eval.evaluate(pred), pr_eval.evaluate(pred)

# COMMAND ----------
# เทียบ 3 algorithm ด้วยตัวแปรครบชุด
import time

algos = {
    "Logistic Regression": LogisticRegression(labelCol="label", weightCol="weight", maxIter=50),
    "Random Forest": RandomForestClassifier(labelCol="label", weightCol="weight", numTrees=50, maxDepth=8, seed=42),
    "GBT": GBTClassifier(labelCol="label", weightCol="weight", maxIter=30, maxDepth=5, seed=42),
}
rows, fitted_models = [], {}
for name, algo in algos.items():
    t0 = time.time()
    fitted, roc, pr = evaluate(build(algo, NUM))
    fitted_models[name] = fitted
    rows.append({"model": name, "ROC-AUC": round(roc, 4), "PR-AUC": round(pr, 4), "train+eval (s)": round(time.time() - t0, 1)})
algo_res = pd.DataFrame(rows)
display(algo_res)

# COMMAND ----------
# Baseline: ตัดตัวแปรออกทีละชุด (Random Forest)
feature_sets = {
    "full": (NUM, True),
    "age_days only": (["age_days"], False),
    "drop age_days": ([c for c in NUM if c != "age_days"], True),
    "drop ios_ver": ([c for c in NUM if c != "ios_ver"], True),
    "drop age_days + ios_ver": ([c for c in NUM if c not in ("age_days", "ios_ver")], True),
}
rows = []
for name, (cols_, genre) in feature_sets.items():
    _, roc, pr = evaluate(build(RandomForestClassifier(labelCol="label", weightCol="weight",
                                                       numTrees=50, maxDepth=8, seed=42), cols_, genre))
    rows.append({"feature set": name, "ROC-AUC": round(roc, 4), "PR-AUC": round(pr, 4)})
display(pd.DataFrame(rows))

# COMMAND ----------
# Feature importance ของ GBT (รวมหมวดทุกค่าเป็นตัวแปรเดียว)
gbt_fit = fitted_models["GBT"]
attrs = gbt_fit.transform(test.limit(1)).schema["features"].metadata["ml_attr"]["attrs"]
names = {a["idx"]: a["name"] for group in attrs.values() for a in group}
imp = gbt_fit.stages[-1].featureImportances.toArray()
imp_s = pd.Series({names[i]: imp[i] for i in range(len(imp))})
imp_s = imp_s.groupby(lambda n: "Primary_Genre" if n.startswith("genre_vec") else n.replace("_imp", "")).sum().sort_values()
imp_s.plot.barh(figsize=(7, 4), title="GBT feature importance (Spark MLlib)")
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC **อ่านผลยังไง**
# MAGIC - ตารางแรก: algorithm ไหน ROC-AUC และ PR-AUC สูงสุด เทียบ PR-AUC กับค่าฐาน (สัดส่วน hit) ที่พิมพ์ไว้ด้านบน
# MAGIC - ตาราง baseline: ถ้า `age_days only` ได้ AUC ใกล้ `full` แปลว่าอายุแอปอธิบายได้เกือบหมด ถ้าห่างกันมาก แปลว่าตัวแปรอื่นช่วยทำนายจริง
# MAGIC - กราฟ importance ของ GBT เทียบกับ permutation importance ของข้อ 9 ได้ ถ้าลำดับต้น ๆ ตรงกัน ผลก็น่าเชื่อขึ้น
# MAGIC - ความสำคัญของตัวแปรคือความสามารถในการทำนาย ไม่ใช่สาเหตุ
# MAGIC
# MAGIC **ผลที่ได้** (ข้อมูล 1,230,315 แอป, hit 1.96%)
# MAGIC - **GBT ดีที่สุด:** ROC-AUC 0.884, PR-AUC 0.199 ซึ่งราว 10 เท่าของค่าฐาน 0.0196 ตามด้วย Random Forest (0.874 / 0.180) และ Logistic Regression (0.852 / 0.125) ใกล้กับโมเดล sklearn ข้อ 9 (AUC 0.895 บนตัวอย่าง 25%)
# MAGIC - **โมเดลไม่ได้แค่รู้อายุแอป:** ใช้ `age_days` อย่างเดียวได้ ROC-AUC แค่ 0.667 พอตัด `age_days` ออกยังได้ 0.828 และตัดทั้ง `age_days` กับ `ios_ver` ยังได้ 0.809
# MAGIC - **ตัวแปรสำคัญของ GBT:** หมวด > ขนาด > อายุแอป > iOS ขั้นต่ำ ส่วนฟรี/เสียเงินแทบไม่ช่วย

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10.3 จัดกลุ่มแอปด้วย k-means (Spark MLlib)
# MAGIC **ทำอะไร:** แบ่งแอปที่มีรีวิวเป็นกลุ่มตามลักษณะ โดยไม่บอกโมเดลว่าแอปไหน hit แล้วดูว่าแต่ละกลุ่มเป็นแอปแบบไหน
# MAGIC
# MAGIC **วัดยังไง:**
# MAGIC - ตัวแปร: log ของรีวิว, ดาว, log ของขนาด, ราคา, อายุแอป, วันที่ไม่ได้อัปเดต, log ของจำนวนแอปของนักพัฒนา แล้วปรับสเกลด้วย `StandardScaler`
# MAGIC - เลือก k จาก **elbow** (cost ลดลงช้าลงตรงไหน) และ **silhouette** (ยิ่งสูงยิ่งแยกกลุ่มชัด) ทดลองบนตัวอย่าง 20% เพื่อประหยัดเวลา แล้วฝึกโมเดลสุดท้ายบนข้อมูลทั้งหมด
# MAGIC - ใช้ log กับตัวแปรที่เบ้มาก ไม่งั้นแอปรีวิวหลักล้านไม่กี่ตัวจะดึงกลุ่มไปหมด
# MAGIC
# MAGIC **อ่านผลยังไง:** ดูตาราง profile ว่าแต่ละกลุ่มมีค่าเฉลี่ยต่างกันตรงไหน แล้วตั้งชื่อ (persona) ให้แต่ละกลุ่ม

# COMMAND ----------
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

km_cols = ["log_reviews", "Average_User_Rating", "log_size", "Price", "age_days", "days_since_update", "log_dev_apps"]
km_df = (clean.where("Reviews > 0 AND release_year BETWEEN 2008 AND 2021")
         .withColumn("log_reviews", F.log1p("Reviews"))
         .withColumn("log_size", F.log1p("size_mb"))
         .withColumn("log_dev_apps", F.log1p("dev_app_count"))
         .withColumn("age_days", F.col("age_days").cast("double"))
         .withColumn("days_since_update", F.col("days_since_update").cast("double"))
         .na.drop(subset=km_cols))
km_prep = Pipeline(stages=[VectorAssembler(inputCols=km_cols, outputCol="raw"),
                           StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)]).fit(km_df)
km_vec = km_prep.transform(km_df)
km_sample = km_vec.sample(fraction=0.2, seed=42)

sil_eval = ClusteringEvaluator(featuresCol="features", predictionCol="prediction", metricName="silhouette")
rows = []
for k in range(2, 9):
    mk = KMeans(featuresCol="features", k=k, seed=42).fit(km_sample)
    rows.append({"k": k, "cost (WSSSE)": mk.summary.trainingCost, "silhouette": sil_eval.evaluate(mk.transform(km_sample))})
k_res = pd.DataFrame(rows)
display(k_res)

fig, ax1 = plt.subplots(figsize=(7, 3.5))
ax1.plot(k_res["k"], k_res["cost (WSSSE)"], "o-")
ax1.set_xlabel("k")
ax1.set_ylabel("cost (WSSSE)")
ax2 = ax1.twinx()
ax2.plot(k_res["k"], k_res["silhouette"], "s--", color="tab:orange")
ax2.set_ylabel("silhouette")
ax1.set_title("Elbow and silhouette")
plt.tight_layout()
plt.show()

# COMMAND ----------
# ตั้ง K ตามกราฟด้านบน (ดูจุดหักของ cost ที่ silhouette ยังสูง) แล้วรันเซลล์นี้
K = 4
km_model = KMeans(featuresCol="features", k=K, seed=42).fit(km_vec)
clustered = km_model.transform(km_vec).withColumnRenamed("prediction", "cluster")
print("silhouette (all data):", round(sil_eval.setPredictionCol("cluster").evaluate(clustered), 4))

profile = (clustered.groupBy("cluster").agg(
    F.count("*").alias("apps"),
    F.round(F.expr("percentile_approx(Reviews, 0.5)"), 0).alias("median_reviews"),
    F.round(F.avg((F.col("Reviews") > 1000).cast("int")) * 100, 1).alias("hit_pct"),
    F.round(F.avg("Average_User_Rating"), 2).alias("avg_rating"),
    F.round(F.expr("percentile_approx(size_mb, 0.5)"), 1).alias("median_size_mb"),
    F.round(F.avg((F.col("Price") > 0).cast("int")) * 100, 1).alias("paid_pct"),
    F.round(F.avg("age_days") / 365, 1).alias("avg_age_years"),
    F.round(F.avg("is_zombie") * 100, 1).alias("zombie_pct"),
    F.round(F.expr("percentile_approx(dev_app_count, 0.5)"), 0).alias("median_dev_apps"),
).orderBy("cluster"))
display(profile)

# COMMAND ----------
# ภาพ 2 มิติด้วย PCA (สุ่ม 5,000 แอปมาวาด)
from pyspark.ml.feature import PCA

pca_model = PCA(k=2, inputCol="features", outputCol="pc").fit(clustered)
print("explained variance:", [round(float(v), 3) for v in pca_model.explainedVariance])
vis = (pca_model.transform(clustered.sample(fraction=min(1.0, 5000 / clustered.count()), seed=1))
       .select("pc", "cluster").toPandas())
vis["x"] = vis["pc"].apply(lambda v: float(v[0]))
vis["y"] = vis["pc"].apply(lambda v: float(v[1]))
fig, ax = plt.subplots(figsize=(7, 5.5))
for c, g in vis.groupby("cluster"):
    ax.scatter(g["x"], g["y"], s=8, alpha=0.5, label=f"cluster {c}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"k-means clusters (k={K}), PCA 2D")
ax.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC **ชื่อของแต่ละกลุ่ม (persona):** เติมหลังเห็นตาราง profile เช่น กลุ่มที่รีวิวสูงและอัปเดตบ่อยอาจเป็น "ดาวรุ่ง" กลุ่มที่ไม่ได้อัปเดตนานอาจเป็น "ซอมบี้"
# MAGIC
# MAGIC **เลือก k = 4:** cost ลดลงมากที่สุดช่วง k = 3 → 4 และ silhouette ยังราว 0.35 ก่อนตกเหลือ 0.22 ที่ k = 7 (silhouette บนข้อมูลทั้งชุด 0.336 = แยกกลุ่มได้ปานกลาง) ใช้เฉพาะแอปที่มีรีวิว
# MAGIC
# MAGIC | cluster | ชื่อ | ลักษณะเด่น |
# MAGIC |---|---|---|
# MAGIC | 1 | **ดาวเด่น** | 76,686 แอป มัธยฐานรีวิว 335, hit 31%, ดาว 4.39, ขนาดใหญ่สุด (มัธยฐาน 101 MB), ร้างแค่ 21.5% |
# MAGIC | 2 | **น้องใหม่** | กลุ่มใหญ่สุด 246,706 แอป อายุเฉลี่ย 2.9 ปี ดาวสูงสุด 4.61 แต่รีวิวยังน้อย (มัธยฐาน 3) |
# MAGIC | 0 | **ดาวร่วง** | 91,314 แอป ดาวเฉลี่ยต่ำสุด 1.86, รีวิวน้อย (มัธยฐาน 2), ร้าง 60% |
# MAGIC | 3 | **ซอมบี้จากโรงงานแอป** | 131,211 แอป อายุเฉลี่ย 7.5 ปี ร้าง 89%, นักพัฒนามีแอปมัธยฐาน 18 แอป, hit 0.3% |
# MAGIC
# MAGIC **อ่านผล:** แอปที่ดังเกือบทั้งหมดอยู่ในกลุ่มดาวเด่นกลุ่มเดียว ซึ่งต่างจากกลุ่มอื่นที่ขนาดใหญ่และยังดูแลอยู่ ส่วนกลุ่มที่ร้างที่สุดมาจากนักพัฒนาที่ทำแอปจำนวนมาก สอดคล้องกับข้อ 7

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10.4 สรุปและคำแนะนำสำหรับนักพัฒนาแอป
# MAGIC **คำถามหลัก:** ล้านแอป แต่มีส่วนน้อยที่มีคนรู้จัก แล้วแอปที่รอดทำอะไรต่างจากแอปอื่น
# MAGIC
# MAGIC | ข้อค้นพบ | มาจาก | คำแนะนำ |
# MAGIC |---|---|---|
# MAGIC | ตลาดกระจุกตัวมาก แอปส่วนน้อยได้รีวิวส่วนใหญ่ และแอปกว่าครึ่งไม่มีรีวิวเลย | ข้อ 1, 2 | ตั้งเป้าให้สมจริง การได้รีวิวแรก ๆ ก็ทำให้ขึ้นมาอยู่ครึ่งบนของตลาดแล้ว |
# MAGIC | แต่ละหมวดมีโอกาสติดตลาดและสัดส่วนแอปร้างต่างกัน | ข้อ 3 | เลือกหมวดจาก `pct_hit` คู่กับ `pct_zombie` ไม่ใช่จากจำนวนแอปในหมวด |
# MAGIC | แอปที่อัปเดตไม่นานติดตลาดมากกว่าชัดเจน | ข้อ 8 | อัปเดตสม่ำเสมอ (เป็นความสัมพันธ์ ยังไม่ใช่สาเหตุ) |
# MAGIC | ขนาดแอปสัมพันธ์กับการติดตลาด | ข้อ 6 | ขนาดสะท้อนความลงทุนกับแอป ไม่ใช่ทำแอปให้ใหญ่ขึ้นแล้วจะดัง |
# MAGIC | ความต่างของดาวระหว่างกลุ่มมี effect size เล็ก แม้ p-value จะเล็กมาก | ข้อ 10 | อย่าตัดสินใจจาก p-value อย่างเดียวเมื่อข้อมูลใหญ่ |
# MAGIC | โมเดล GBT ทำนาย hit ได้ ROC-AUC 0.884 แต่ใช้อายุแอปอย่างเดียวได้แค่ 0.667 ตัวแปรสำคัญคือหมวด ขนาด อายุ | ข้อ 10.2 | ความสำเร็จไม่ได้มาจากการอยู่มานานอย่างเดียว การเลือกหมวดและการลงทุนกับตัวแอปมีความสัมพันธ์ชัด |
# MAGIC | k-means แบ่งแอปที่มีรีวิวเป็น 4 กลุ่ม "ดาวเด่น" (hit 31%) แยกจากกลุ่มอื่นชัด ส่วน "ซอมบี้จากโรงงานแอป" ร้าง 89% | ข้อ 10.3 | ทำแอปน้อยแต่ดูแลต่อเนื่อง ดีกว่าผลิตจำนวนมากแล้วปล่อยทิ้ง |
# MAGIC
# MAGIC **ข้อควรระวัง:** ทุกข้อเป็นความสัมพันธ์จาก snapshot ครั้งเดียว ไม่ใช่เหตุและผล (ดูข้อ 12)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. ตารางสรุปสำหรับทำเว็บ
# MAGIC **ทำอะไร:** ยุบข้อมูล 1.2 ล้านแถวเป็นตารางสรุปขนาดเล็ก (ราวหมื่นแถว) ให้หน้าเว็บโหลดและกรองได้เร็วโดยไม่ต้องต่อ Databricks
# MAGIC
# MAGIC **วัดยังไง:** จัดกลุ่มตามปีเปิดตัว, หมวด, ฟรี/เสียเงิน, ช่วงขนาด และขนาดนักพัฒนา แล้วเก็บผลรวมของจำนวนแอป แอปที่มีรีวิว แอปที่ติดตลาด แอปร้าง รีวิวรวม และผลรวมดาว หน้าเว็บคำนวณเปอร์เซ็นต์เองจากผลรวมเหล่านี้
# MAGIC
# MAGIC **วิธีดาวน์โหลด:** กดไอคอนดาวน์โหลดที่ผลลัพธ์ เลือก **Download full results** (ถ้าเลือกแบบแสดงผล จะได้แค่ 10,000 แถวแรก)

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT release_year, Primary_Genre, Free,
# MAGIC        CASE WHEN size_mb IS NULL THEN 'unknown'
# MAGIC             WHEN size_mb < 10  THEN '<10 MB'
# MAGIC             WHEN size_mb < 50  THEN '10-50 MB'
# MAGIC             WHEN size_mb < 100 THEN '50-100 MB'
# MAGIC             WHEN size_mb < 500 THEN '100-500 MB'
# MAGIC             ELSE '500+ MB' END AS size_bucket,
# MAGIC        CASE WHEN dev_app_count = 1   THEN 'indie'
# MAGIC             WHEN dev_app_count <= 10  THEN '2-10 apps'
# MAGIC             WHEN dev_app_count <= 100 THEN '11-100 apps'
# MAGIC             ELSE '100+ apps' END AS dev_size,
# MAGIC        COUNT(*) AS apps,
# MAGIC        SUM(CASE WHEN Reviews > 0 THEN 1 ELSE 0 END) AS with_reviews,
# MAGIC        SUM(CASE WHEN Reviews > 1000 THEN 1 ELSE 0 END) AS hits,
# MAGIC        SUM(is_zombie) AS zombies,
# MAGIC        SUM(Reviews) AS total_reviews,
# MAGIC        SUM(CASE WHEN Reviews > 0 THEN Average_User_Rating END) AS rating_sum
# MAGIC FROM apps
# MAGIC WHERE release_year BETWEEN 2008 AND 2021
# MAGIC GROUP BY ALL;

# COMMAND ----------
# MAGIC %md
# MAGIC ## 12. ข้อจำกัดของการวิเคราะห์
# MAGIC - ข้อมูลเป็น **snapshot** ครั้งเดียว ไม่มีประวัติการเปลี่ยนแปลงของแอปแต่ละตัว จึงบอกไม่ได้ว่าการเปลี่ยนแปลงใดเป็นสาเหตุ
# MAGIC - ไม่มีข้อมูลอันดับ การถูก feature หรือ algorithm ของ Apple จึงสรุปเรื่อง algorithm ไม่ได้
# MAGIC - ไม่มียอดดาวน์โหลด ใช้จำนวนรีวิวเป็นตัวแทนความนิยม ซึ่งไม่เท่ากับจำนวนผู้ใช้
# MAGIC - แอปรุ่นใหม่มีเวลาสะสมรีวิวน้อยกว่าแอปรุ่นเก่า การเทียบข้ามรุ่นปีจึงต้องระวัง
# MAGIC - ผลทั้งหมดเป็นความสัมพันธ์ ไม่ใช่ความเป็นเหตุเป็นผล
# MAGIC - `Current_Version_Score` ไม่ต่างจาก `Average_User_Rating` ในข้อมูลนี้ จึงสร้างตัวแปรแนวโน้มคะแนนไม่ได้
# MAGIC - `Current_Version_Reviews` เท่ากับ `Reviews` ทุกแถว จึงแยกไม่ได้ว่าแอปยังได้รีวิวใหม่อยู่ หรือเคยดังแล้วเงียบไป
# MAGIC - **วันอ้างอิงน่าจะช้ากว่าความจริง:** `ref_date` = `max(Updated)` = 2022-04-09 แต่มีแอปที่อัปเดตหลังกลาง ต.ค. 2021 แค่ 288 แอป และไฟล์ลงวันที่ 14 ต.ค. 2021 วันเก็บข้อมูลจริงจึงน่าจะราวกลาง ต.ค. 2021 ผลคือ `age_days` และ `days_since_update` สูงกว่าจริงราว 6 เดือนทุกแอป เกณฑ์แอปร้าง 730 วันเทียบกับวันจริงจึงเท่ากับไม่อัปเดตราว 1.5 ปี notebook นี้คงค่าเดิมไว้เพื่อให้ผลทุกหัวข้อเทียบกันได้
