# install necessary libs
echo "install necessary libs"
yum install -y git
yum install -y cmake
yum install -y gcc-c++
 
# deploy
echo "git clone xgboost and deploy"
git clone --recursive https://github.com/dmlc/xgboost
cd xgboost && git checkout 76c320e9f0db7cf4aed73593ddcb4e0be0673810
cd ${HOME}/xgboost/dmlc-core && git checkout 706f4d477a48fc75cb46b226ea007fbac862f9c2
cd ${HOME}/xgboost/rabit && git checkout 112d866dc92354304c0891500374fe40cdf13a50
sed '/USE_HDFS/s/0/1/g' ${HOME}/xgboost/make/config.mk > ${HOME}/xgboost/config.mk

cd ${HOME}/xgboost  && make -j22
mv ${HOME}/xgboost /home/hadoop/
chown -R hadoop:hadoop /home/hadoop/xgboost

NAMENODE_AND_PORT=$(hdfs getconf -confKey fs.defaultFS)
echo "NAMENODE_AND_PORT: ${NAMENODE_AND_PORT}"
HADOOP_CONF_DIR=${HADOOP_HOME}/etc/hadoop
cat << CORE_SITE_CONF_EOF > ${HADOOP_CONF_DIR}/core-site.xml
<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="configuration.xsl"?>
<!--
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License. See accompanying LICENSE file.
-->

<!-- Put site-specific property overrides in this file. -->

<configuration>
  <property>
    <name>fs.defaultFS</name>
    <value>${NAMENODE_AND_PORT}/</value>
  </property>
</configuration>
CORE_SITE_CONF_EOF

# test
echo "set xgboost configurations"
TEST_DIR=/home/hadoop/xgboost-test
mkdir -p ${TEST_DIR}
cat << XGBOOST_CONF_EOF > ${TEST_DIR}/mushroom.hadoop.conf
booster = gbtree
objective = binary:logistic
save_period = 0
dsplit=row
eval_train = 1
eval_metric = auc
silent = 1
XGBOOST_CONF_EOF

echo "write test script"
cat << TEST_SCRIPT_EOF > ${TEST_DIR}/test_xgboost.sh
#!/bin/bash

XGBOOST_HOME=/home/hadoop/xgboost
HDFS_PREFIX=${NAMENODE_AND_PORT}
export HADOOP_CONF_DIR=\$HADOOP_HOME/etc/hadoop
# set env value of HADOOP_HDFS_HOME which is needed by run_hdfs_prog.py
export HADOOP_HDFS_HOME=\${HADOOP_HOME}
# number of workers
NUM_WORKERS=2
# number of threads in ecah worker
NUM_THREADS=2

#put the local training file to HDFS
hadoop fs -test -d /user/hadoop/xgboost-test-data || hadoop fs -mkdir -p /user/hadoop/xgboost-test-data
hadoop fs -test -e /user/hadoop/xgboost-test-data/agaricus.txt.train || hadoop fs -put \${XGBOOST_HOME}/demo/data/agaricus.txt.train /user/hadoop/xgboost-test-data
hadoop fs -test -e /user/hadoop/xgboost-test-data/agaricus.txt.test || hadoop fs -put \${XGBOOST_HOME}/demo/data/agaricus.txt.test /user/hadoop/xgboost-test-data

# running rabit, pass address in hdfs
\${XGBOOST_HOME}/dmlc-core/tracker/dmlc_yarn.py  -n \${NUM_WORKERS} --vcores \${NUM_THREADS} \${XGBOOST_HOME}/xgboost ${TEST_DIR}/mushroom.hadoop.conf nthread=\${NUM_THREADS}\
    data=\${HDFS_PREFIX}/user/hadoop/xgboost-test-data/agaricus.txt.train\
    eval[test]=\${HDFS_PREFIX}/user/hadoop/xgboost-test-data/agaricus.txt.test\
    model_out=\${HDFS_PREFIX}/user/hadoop/xgboost-test-data/mushroom.final.model\
    max_depth=3\
    num_round=10 1>${TEST_DIR}/yarn-info

#get the final model file
[ -e ${TEST_DIR}/final.model ] && rm -rf ${TEST_DIR}/final.model
hadoop fs -get \${HDFS_PREFIX}/user/hadoop/xgboost-test-data/mushroom.final.model ${TEST_DIR}/final.model

# use dmlc-core/yarn/run_hdfs_prog.py to setup approperiate env
# output prediction task=pred
\${XGBOOST_HOME}/dmlc-core/yarn/run_hdfs_prog.py \${XGBOOST_HOME}/xgboost ${TEST_DIR}/mushroom.hadoop.conf task=pred model_in=${TEST_DIR}/final.model test:data=\${XGBOOST_HOME}/demo/data/agaricus.txt.test
# print the boosters of final.model in dump.raw.txt
\${XGBOOST_HOME}/dmlc-core/yarn/run_hdfs_prog.py \${XGBOOST_HOME}/xgboost ${TEST_DIR}/mushroom.hadoop.conf task=dump model_in=${TEST_DIR}/final.model name_dump=${TEST_DIR}/dump.raw.txt
# use the feature map in printing for better visualization
\${XGBOOST_HOME}/dmlc-core/yarn/run_hdfs_prog.py \${XGBOOST_HOME}/xgboost ${TEST_DIR}/mushroom.hadoop.conf task=dump model_in=${TEST_DIR}/final.model fmap=\${XGBOOST_HOME}/demo/data/featmap.txt name_dump=${TEST_DIR}/dump.nice.txt
cat ${TEST_DIR}/dump.nice.txt
[ -e ${TEST_DIR}/yarn-info ] && rm -rf ${TEST_DIR}/yarn-info
TEST_SCRIPT_EOF
chown -R hadoop:hadoop ${TEST_DIR}

echo "test beginning"
bash ${TEST_DIR}/test_xgboost.sh
echo "test end"
