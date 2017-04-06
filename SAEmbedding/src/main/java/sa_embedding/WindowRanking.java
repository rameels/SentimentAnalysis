package sa_embedding;

import java.util.Random;

import duyuNN.*;

/*
 * This class implements the window ranking approach which uses sentiment of sentences 
 * for learning sentiment embedding
 * This is suitable for binary class situation.
 */
public class WindowRanking {
	public LookupLayer lookup;
	public LinearLayer linear1;
	public TanhLayer htanh;
	public LinearLayer linear2;
	
	public WindowRanking()
	{
	}
	
	public int[] input;
	public double[] output;
	public double[] outputG;
	
	public int windowSize;
	public int vocabSize;
	public int hiddenSize;
	public int embeddingLength;
	
	public WindowRanking(
			int xWindowSize,
			int xVocabSize,
			int xHiddenSize,
			int xEmbeddingLength) throws Exception
	{
		windowSize = xWindowSize;
		vocabSize = xVocabSize;
		hiddenSize = xHiddenSize;
		embeddingLength = xEmbeddingLength;
		
		lookup = new LookupLayer(embeddingLength, vocabSize, windowSize);
		linear1 = new LinearLayer(windowSize * embeddingLength, hiddenSize);
		htanh = new TanhLayer(hiddenSize);
		linear2 = new LinearLayer(hiddenSize, 2);
		
		lookup.link(linear1);
		linear1.link(htanh);
		htanh.link(linear2);
		
		input = lookup.input;
		output = linear2.output;
		outputG = linear2.outputG;
	}
	
	public void randomize(Random rnd, double min, double max)
	{
		lookup.randomize(rnd, min, max);
		linear1.randomize(rnd, min/linear1.inputLength, max/linear1.inputLength);
		linear2.randomize(rnd, min/linear2.inputLength, max/linear2.inputLength);
	}
	
	public void forward()
	{
		lookup.forward();
		linear1.forward();
		htanh.forward();
		linear2.forward();
	}
	
	public void backward()
	{
		linear2.backward();
		htanh.backward();
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
		htanh.clearGrad();
		linear2.clearGrad();
	}
}
