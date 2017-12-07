from pyspark import SparkContext, SparkConf
import sys
from operator import itemgetter
import os
os.environ["SPARK_HOME"] = "/usr/local/qqwebsrv/spark-1.6.1"


if len(sys.argv) < 5:
    print "<Usage>: input_path model_file start_id output_path"
    sys.exit(1)
input_path = sys.argv[1]
model_file = sys.argv[2]
start_id = int(sys.argv[3])
output_path = sys.argv[4]

# Load dumped model first
trees = []
tree = {}
trees_id = []
tree_id = {}
class Node(object):
    def __init__(self, value, fid=None, yes=None, no=None, missing=None, is_leaf=False):
        self.value = value
        self.fid = fid
        self.yes = yes
        self.no = no
        self.missing = missing
        self.is_leaf = is_leaf

with open(model_file) as f:
    for line in f:
        if ("booster" in line):
            if len(tree)>1:
                trees.append(tree)
                trees_id.append(tree_id)
            tree = {}
            tree_id = {}
            continue
        line = line.strip()
        nodeid = int(line.split(':')[0])
        if ("leaf" in line):
            value = line.split('=')[1]
            tree[nodeid] = Node(float(value), nodeid, is_leaf=True)
            tree_id[nodeid] = start_id
            start_id += 1
        else:
            fid = line.split('f')[1].split('<')[0]
            value = line.split('<')[1].split(']')[0]
            yes = line.split('yes=')[1].split(',')[0]
            no = line.split('no=')[1].split(',')[0]
            missing = line.split('missing=')[1]
            tree[nodeid] = Node(float(value), fid, int(yes), int(no), int(missing))
    if len(tree)>1:
        trees.append(tree)
        trees_id.append(tree_id)


def get_leaf_id(tree, feats):
    node = tree[0]
    while(not node.is_leaf):
        if node.fid in feats:
            if(feats[node.fid] < node.value):
                node = tree[node.yes]
            else:
                node = tree[node.no]
        else:
            node = tree[node.missing]
    return node.fid
        

# Now do predict
def mapper_pred(line):
    line = line.strip('\n')
    lines = line.split()
    feats = {}
    for i in range(1, len(lines)):
        fid, fvalue = lines[i].split(':')
        feats[fid] = float(fvalue)

    # Now comp tree
    for (tree, tree_id) in zip(trees, trees_id):
        leaf_id = get_leaf_id(tree, feats)
        line = line + "\t" + str(tree_id[leaf_id]) + ":1"
    return line
   

sc = SparkContext()
res = sc.textFile(input_path).map(mapper_pred).saveAsTextFile(output_path)
