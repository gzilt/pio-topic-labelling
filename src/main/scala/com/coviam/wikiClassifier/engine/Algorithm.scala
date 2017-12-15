package com.coviam.wikiClassifier.engine

import org.apache.predictionio.controller.{P2LAlgorithm, Params, PersistentModel, PersistentModelLoader}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{HashingTF}
import org.apache.spark.ml.classification.{NaiveBayes,NaiveBayesModel}
import org.apache.spark.sql._



class Algorithm(val ap:AlgorithmParams) extends P2LAlgorithm[PreparedData, Model, Query, PredictedResult]{

  override def train(sc: SparkContext, pd: PreparedData): Model = {

    val nbModel: NaiveBayesModel =  new NaiveBayes().fit(pd.labeledPoints)
    val spark = SparkSession.builder().config(sc.getConf).getOrCreate()
    import spark.implicits._
    val obs = pd.trainingData.contentAndcategory
    //val sqlContext = SQLContext.getOrCreate(sc)
    val phraseDataFrame = spark.createDataFrame(obs).toDF("content", "category")
    val categories: Map[String,Int] = phraseDataFrame.map(row => row.getAs[String]("category")).collect().zipWithIndex.toMap
    Model(nbModel, categories, sc)
  }

  override def predict(model: Model, query: Query): PredictedResult = {
    val qryInd = query.topics.zipWithIndex
    val spark = SparkSession.builder().config(model.sc.getConf).getOrCreate()
    val df = spark.createDataFrame(qryInd).toDF("words","id")
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("features")
    val featurizedData = hashingTF.transform(df)
    val featureSet = featurizedData.select("features")
    val categories = model.categories.map(_.swap)
    val prediction = model.nbModel.transform(featureSet).select("prediction").first().getDouble(0).toInt
    val cat = categories(prediction)
    PredictedResult(cat)
  }
}
case class Model( nbModel: NaiveBayesModel,
                  categories : Map[String,Int],
                  sc: SparkContext) extends PersistentModel[AlgorithmParams] with Serializable {

  def save(id: String, params: AlgorithmParams, sc: SparkContext): Boolean = {
    nbModel.save(s"/tmp/${id}/nbmodel")
    sc.parallelize(Seq(categories)).saveAsObjectFile(s"/tmp/${id}/categories")
    true
  }

}

object Model extends PersistentModelLoader[AlgorithmParams, Model]{
  def apply(id: String, params: AlgorithmParams, sc: Option[SparkContext]) = {
   // println(sc.get.objectFile(s"/tmp/${id}/categories"))
    new Model(
      NaiveBayesModel.load(s"/tmp/${id}/nbmodel"),
      sc.get.objectFile[Map[String,Int]](s"/tmp/${id}/categories").first,
      sc.get
    )
  }
}

case class AlgorithmParams(val lambda:Double) extends Params
