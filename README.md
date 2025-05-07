# Step 1 : Read File
df = spark.read.format("csv").option("header", "true").load("dbfs:/FileStore/shared_uploads/rupanjalisingh10@gmail.com/data__1_.csv")
df.display()

## Step 2: Data Exploration and Preprocessing
from pyspark.sql.functions import col, isnan, when, count

df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()
## Step 3: Feature Selection
from pyspark.sql.functions import col

# Convert columns to integer type
numeric_df = df.select(
    col("footfall").cast("int"),
    col("tempMode").cast("int"),
    col("AQ").cast("int"),
    col("USS").cast("int"),
    col("CS").cast("int"),
    col("VOC").cast("int"),
    col("RP").cast("int"),
    col("IP").cast("int"),
    col("Temperature").cast("int"),
    col("fail").cast("int")
)

from pyspark.ml.feature import VectorAssembler

# Assemble features into a single vector
feature_columns = ['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP', 'Temperature']
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
vector_df = assembler.transform(numeric_df)
## Step 4: Model Building
# Split the data into training and test sets
train_df, test_df = vector_df.randomSplit([0.8, 0.2], seed=42)

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(featuresCol='features', labelCol='fail', numTrees=100)

model = rf.fit(train_df)
## Step 5: Model Evaluation
# Make predictions on the test set
predictions = model.transform(test_df)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize the evaluator
evaluator = MulticlassClassificationEvaluator(labelCol="fail", predictionCol="prediction", metricName="accuracy")

# Calculate accuracy
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")
## Step 6: Model Tuning
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create a parameter grid for tuning
paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [50, 100, 150])
             .build())

# Set up cross-validation
crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

# Run cross-validation
cvModel = crossval.fit(train_df)
### Step 7: Deployment
# Save the trained model
model.save("/path/to/save/model")
