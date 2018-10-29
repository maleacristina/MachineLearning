package wekalinearregression;


import weka.classifiers.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.functions.LinearRegression;

public class WekaLinearRegression{
    
	public static void main(String args[]) throws Exception{
		//Load Data set
		DataSource source = new DataSource("C:\\Users\\Asus\\Desktop\\Java_Programs\\WekaLinearRegression\\src\\wekalinearregression\\house.arff");
		Instances dataset = source.getDataSet();
		//set class index to the last attribute
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		//Build model
		LinearRegression model = new LinearRegression();
		model.buildClassifier(dataset);
		//output model
		System.out.println("LR FORMULA : " + model);	
		
		// Now Predicting the cost 
		Instance house = dataset.lastInstance();
		double price = model.classifyInstance(house);
		System.out.println("-------------------------");	
		System.out.println("PRECTING THE PRICE : " + price);	
                Evaluation eval=new Evaluation(dataset);
                eval.evaluateModel(model,dataset);
                System.out.println(eval.toSummaryString());
	}
}