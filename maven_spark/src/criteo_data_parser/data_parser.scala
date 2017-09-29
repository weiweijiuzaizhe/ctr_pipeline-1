package criteo_data_parser

import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer

/**
  * Created by weiwei on 2017/9/28.
  */
object data_parser {
  def main(args: Array[String]): Unit = {

    val mode_flag = args(0)
    val min_bar = args(1).toInt
    println("min_bar:" + min_bar)
    val file_input = args(2)
    println("file_input:" + file_input)
    val file_output = args(3)


    val conf = new SparkConf() //??SparkConf??
    conf.setAppName("criteo_parser") //?????????,????????????????
    if( mode_flag=="test"){
      conf.setMaster("local") //?????????,?????Spark??
    }


    val sc = new SparkContext(conf) //??SparkContext,????SparkConf?????Spark????????????

    val lines = sc.textFile(file_input) //??????,??????Partition

    val num_split_res = lines.map(line => line.split("\t")).flatMap {
      arr =>
        val l = ListBuffer[String]()
        for (i <- 1 until arr.length) {
          l.append(arr(i).trim() + "-" + i)
        }
        l
    }

    //TODO:?????????
    val filter_wc = num_split_res.map {
      i => (i, 1)
    }.reduceByKey(_ + _).filter(i => i._2 >= min_bar).map(i => i._1).collect()


    var i = 0
    val hs = new scala.collection.mutable.HashMap[String, Int]
    for (item <- filter_wc) {
      i = i + 1
      hs(item) = i //????kv?
    }

    val parse_res = lines.map {
      line =>
        val e = line.split("\t")
        val filter_l = ListBuffer[String]()
        for (i <- 1 until e.length) {
          val key = e(i) + "-" + i
          val value = hs.getOrElse(key, -1)
          if (value > 0) {
            filter_l.append(value + ":1")
          }
        }
        e(0) + " " + filter_l.mkString(" ")  //?????
    }



    if(mode_flag == "test"){
      parse_res.collect().foreach(println)
    }
    if(mode_flag == "cluster"){
      parse_res.saveAsTextFile(file_output)
    }



    sc.stop()

  }
}

