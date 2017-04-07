package runsvm;

import java.io.IOException;

import de.bwaldvogel.liblinear.InvalidInputDataException;
import de.bwaldvogel.liblinear.Predict;
import de.bwaldvogel.liblinear.Train;

public class LibLinearUsage {

	public static void train(
			String trainFile, 
			String c, // c is -c in LibLinear
			String modelFile) 
			throws IOException, InvalidInputDataException
	{
		Train train = new Train();
		String[] trainArgs = {"-c", c, trainFile, modelFile};
		
		train.main(trainArgs);
	}
	
	public static void predict(
			String modelFile, 
			String testFile, 
			String outputFile) throws IOException
	{
		Predict predict = new Predict();
		String[] predictArgs = {testFile, modelFile, outputFile};
		
		predict.main(predictArgs);
	}
}
