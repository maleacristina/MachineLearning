package clustering;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.clusterers.EM;
import weka.clusterers.HierarchicalClusterer;


public class Clustering {
    public static void main(String[] args) throws Exception{
	DataSource source = new DataSource("vote.arff");
	Instances traindata = source.getDataSet();
			
	SimpleKMeans kmeans = new SimpleKMeans();
	kmeans.setNumClusters(4);
	kmeans.buildClusterer(traindata);
		
	//em
	EM model = new EM();
	// build the Weka clusterer
	model.buildClusterer(traindata);
	ClusterEvaluation eval = new ClusterEvaluation();
	eval.setClusterer(model);
	eval.evaluateClusterer(traindata);
	System.out.println(eval.clusterResultsToString());
		
	HierarchicalClusterer hierarchicalClusterer = new HierarchicalClusterer();
        hierarchicalClusterer.setNumClusters(3);
        hierarchicalClusterer.buildClusterer(traindata);
        eval.setClusterer(hierarchicalClusterer);
        eval.evaluateClusterer(traindata);
        System.out.println(eval.clusterResultsToString());
		
	//ClusterEvaluation eval = new ClusterEvaluation();
	eval.setClusterer(kmeans);
	eval.evaluateClusterer(traindata);
	System.out.println(eval.clusterResultsToString());
    }
}