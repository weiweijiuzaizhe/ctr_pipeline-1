#!/bin/sh

source /etc/profile
HADOOP=/data/clusterserver/hadoop/bin/hadoop


if [ $# -eq 4 ];then
    sample_in=$1
    model_file=$2
	start_id=$3
    sample_out=$4
else
   echo "Usage: <sample_in> <dumped model_file> <start id> <sample_out>"
   exit 0
fi

function xgb_feature() 
{
$HADOOP fs -rmr $sample_out
spark-submit \
 --master yarn \
 --name xgb_feature \
 --num-executors 50 \
 --executor-memory 1G \
 --executor-cores 3 \
 --driver-memory 1G \
 --conf spark.driver.maxResultSize=2G \
 --conf spark.executor.extraJavaOptions="-XX:+UseG1GC -XX:+UseStringDeduplication" \
 --conf spark.serializer=org.apache.spark.serializer.KryoSerializer \
 --conf spark.kryoserializer.buffer.max=512m \
 --conf spark.python.worker.reuse=true \
 --deploy-mode client \
 xgb_feature.py $sample_in $model_file $start_id $sample_out
}

xgb_feature
