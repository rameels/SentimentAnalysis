package sa_embedding;
import java.util.Random;

import duyuNN.*;
/*
 * This class implements the window prediction approach which uses sentiment of sentences 
 * for learning sentiment embedding.
 */
public class WindowPrediction implements NNInterface{
	
	public LookupLayer lookup;
	public LinearLayer linear1;
	public TanhLayer tanh;
	public LinearLayer linear2;
	public SoftmaxLayer softmax;
	
	public WindowPrediction()
	{ 
		
	}
	
	public int[] input;
	public double[] output;
	public double[] outputG;
	
	public int windowSize;
	public int vocabSize;
	public int hiddenSize;
	public int embeddingLength;
	
	public int classNum;
	
	public WindowPrediction(
			int xWindowSize,
			int xVocabSize,
			int xHiddenSize,
			int xEmbeddingLength,
			int xClassNum) throws Exception
	{
		windowSize = xWindowSize;
		vocabSize = xVocabSize;
		hiddenSize = xHiddenSize;
		embeddingLength = xEmbeddingLength;
		classNum = xClassNum;
		
		lookup = new LookupLayer(embeddingLength, vocabSize, windowSize);
		linear1 = new LinearLayer(windowSize * embeddingLength, hiddenSize);
		tanh = new TanhLayer(hiddenSize);
		linear2 = new LinearLayer(hiddenSize, classNum);
		softmax = new SoftmaxLayer(classNum);
		
		lookup.link(linear1);
		linear1.link(tanh);
		tanh.link(linear2);
		linear2.link(softmax);
		
		input = lookup.input;
		output = softmax.output;
		outputG = softmax.outputG;
	}
	
	public void forward()
	{
		lookup.forward();
		linear1.forward();
		tanh.forward();
		linear2.forward();
		softmax.forward();
	}
	
	public void backward()
	{
		softmax.backward();
		linear2.backward();
		tanh.backward();
		linear1.backward();
		lookup.backward();
	}
	
	public void update(double learningRate)
	{
		lookup.update(learningRate);
		linear1.update(learningRate / linear1.inputLength);
		linear2.update(learningRate / linear2.inputLength);
	}
	
	public void clearGrad()
	{
		lookup.clearGrad();
		linear1.clearGrad();
		tanh.clearGrad();
		linear2.clearGrad();
		softmax.clearGrad();
	}

	@Override
	public void randomize(Random r, double min, double max) {
		lookup.randomize(r, min, max);
		linear1.randomize(r, min / linear1.inputLength, max/linear1.inputLength);
		linear2.randomize(r, min/linear2.inputLength, max/linear2.inputLength);
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		return null;
	}
}
