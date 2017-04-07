package runsvm;

import java.io.IOException;
import java.util.HashMap;

import dataprepare.Funcs;
import de.bwaldvogel.liblinear.InvalidInputDataException;
import evaluationMetric.Metric;

public class SVMMain {

	public static void main(String[] args) {
		
		HashMap<String, String> argsMap = Funcs.parseArgs(args);
		
		for(String key: argsMap.keySet())
		{
			System.out.println(key + "\t" + argsMap.get(key));
		}
		
		String trainFile = argsMap.get("-trainFeatureFile");
		String testFile = argsMap.get("-testFeatureFile"); 
		String c = argsMap.get("-c");
		
		String modelFile = trainFile + c + ".model";
		String outputFile = testFile + c + ".out";
		
		try {
			LibLinearUsage.train(trainFile, c, modelFile);
			LibLinearUsage.predict(modelFile, testFile, outputFile);
			
			Metric.calcMetric(testFile, outputFile);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InvalidInputDataException e) {
			e.printStackTrace();
		}
		System.out.println("===============================");
	}

}
