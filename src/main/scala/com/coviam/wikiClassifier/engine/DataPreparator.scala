package com.coviam.wikiClassifier.engine

import org.apache.predictionio.controller.PPreparator
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{Tokenizer, _}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector


class DataPreparator() extends PPreparator[TrainingData, PreparedData]{

  override def prepare(sc: SparkContext, trainingData: TrainingData): PreparedData = {

    val obs = trainingData.contentAndcategory
    val spark = SparkSession.builder().config(sc.getConf).getOrCreate()
    import spark.implicits._
    val phraseDataFrame = spark.createDataFrame(obs).toDF("content", "category")
    val categories: Map[String,Int] = phraseDataFrame.map(row => row.getAs[String]("category")).collect().zipWithIndex.toMap
    val tf = processPhrase(phraseDataFrame)
    val labeledPoints: Dataset[LabeledPoint] = tf.map(row => {
      LabeledPoint(categories(row.getAs[String]("category")).toDouble, row.getAs[Vector]("rowFeatures"))
    })

    PreparedData(trainingData,labeledPoints)
  }

  def processPhrase(phraseDataFrame:DataFrame): DataFrame ={

    val tokenizer = new Tokenizer_new().setInputCol("content").setOutputCol("unigram")
    val unigram = tokenizer.transform(phraseDataFrame)

    val remover = new StopWordsRemover().setInputCol("unigram").setOutputCol("filtered")
    val stopRemoveDF = remover.transform(unigram)

    val htf = new HashingTF().setInputCol("filtered").setOutputCol("rowFeatures")
    val tf = htf.transform(stopRemoveDF)

    tf
  }
}

case class PreparedData(var trainingData: TrainingData, var labeledPoints:Dataset[LabeledPoint]) extends Serializable{
}

class Tokenizer_new extends Tokenizer(){

  override def createTransformFunc: (String) => Seq[String] = { str =>
    val unigram = str.replaceAll("[.*|!*|?*|=*|)|(]","").replaceAll("((www\\.[^\\s]+)|(https?://[^\\s]+)|(http?://[^\\s]+))","")
      .replaceAll("(0-9*)|(0-9)+(A-Za-z)*(.*|:*)","").toLowerCase().split("\\s+").toSeq
    unigram
  }
}
