package sa_embedding;

import duyuNN.*;

import java.util.Random;

import duyuNN.*;
/*
 * This class implements the context prediction approach which uses contexts of words 
 * for enhancing word embedding
 */
public class ContextRanking {

	public LookupLayer lookup;
	public LinearLayer linear1;
	public TanhLayer tanh;
	public LinearLayer linear2;
	
	public ContextRanking()
	{
		
	}
	
	public int[] input;
	public double[] output;
	public double[] outputG;
	
	public int windowSize;
	public int vocabSize;
	public int hiddenSize;
	public int embeddingLength;
	
	public ContextRanking(
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
		tanh = new TanhLayer(hiddenSize);
		linear2 = new LinearLayer(hiddenSize, 1);
		
		lookup.link(linear1);
		linear1.link(tanh);
		tanh.link(linear2);
		
		input = lookup.input;
		output = linear2.output;
		outputG = linear2.outputG;
	}
	
	public ContextRanking cloneWithTiedParams() throws Exception
	{
		ContextRanking clone = new ContextRanking();
		
		clone.windowSize = windowSize;
		clone.vocabSize = vocabSize;
		clone.hiddenSize = hiddenSize;
		clone.embeddingLength = embeddingLength;
		
		clone.lookup = (LookupLayer) lookup.cloneWithTiedParams();
		clone.linear1 = (LinearLayer) linear1.cloneWithTiedParams();
		clone.tanh = (TanhLayer) tanh.cloneWithTiedParams();
		clone.linear2 = (LinearLayer) linear2.cloneWithTiedParams();
		
		clone.lookup.link(clone.linear1);
		clone.linear1.link(clone.tanh);
		clone.tanh.link(clone.linear2);
		
		clone.input = clone.lookup.input;
		clone.output = clone.linear2.output;
		clone.outputG = clone.linear2.outputG;
		
		return clone;
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
		tanh.forward();
		linear2.forward();
	}
	
	public void backward()
	{
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
	}
}
