package sa_embedding;

import java.util.Random;

import duyuNN.*;

/*
 * This class implements the window ranking approach which uses sentiment of sentences 
 * for learning sentiment embedding
 * This is suitable for binary class situation.
 */
public class HybridRanking{
	public LookupLayer lookup;
	public LinearLayer linear1;
	public TanhLayer tanh;
	public LinearLayer sentimentLinear2;
	
	public LinearLayer contextLinear2;
	
	public HybridRanking()
	{
	}
	
	public int[] input;
	
	public int windowSize;
	public int vocabSize;
	public int hiddenSize;
	public int embeddingLength;
	
	public HybridRanking(
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
		sentimentLinear2 = new LinearLayer(hiddenSize, 2);
		
		lookup.link(linear1);
		linear1.link(tanh);
		tanh.link(sentimentLinear2);
		
		input = lookup.input;
		
		contextLinear2 = new LinearLayer(hiddenSize, 1);
		// link tanh to contextLinear2 manually in forward and backward.
	}
	
	public void randomize(Random rnd, double min, double max)
	{
		lookup.randomize(rnd, min, max);
		linear1.randomize(rnd, min/linear1.inputLength, max/linear1.inputLength);
		sentimentLinear2.randomize(rnd, min/sentimentLinear2.inputLength, max/sentimentLinear2.inputLength);
		
		contextLinear2.randomize(rnd, min/contextLinear2.inputLength, max/contextLinear2.inputLength);
	}
	
	public void forward()
	{
		lookup.forward();
		linear1.forward();
		tanh.forward();
		sentimentLinear2.forward();
		
		System.arraycopy(tanh.output, 0, contextLinear2.input, 0, hiddenSize);
		contextLinear2.forward();
	}
	
	public void backward()
	{
		contextLinear2.backward();
		sentimentLinear2.backward();
		
		for(int i = 0; i < hiddenSize; i++)
		{
			tanh.outputG[i] += contextLinear2.inputG[i];
		}
		
		tanh.backward();
		linear1.backward();
		lookup.backward();
	}
	
	public void update(double learningRate)
	{
		lookup.update(learningRate);
		linear1.update(learningRate / linear1.inputLength);
		sentimentLinear2.update(learningRate / sentimentLinear2.inputLength);
		contextLinear2.update(learningRate / contextLinear2.inputLength);
	}
	
	public void clearGrad()
	{
		lookup.clearGrad();
		linear1.clearGrad();
		tanh.clearGrad();
		sentimentLinear2.clearGrad();
		contextLinear2.clearGrad();
	}
	
	public HybridRanking cloneWithTiedParams() throws Exception
	{
		HybridRanking clone = new HybridRanking();
		clone.windowSize = windowSize;
		clone.vocabSize = vocabSize;
		clone.hiddenSize = hiddenSize;
		clone.embeddingLength = embeddingLength;
		
		clone.lookup = (LookupLayer) lookup.cloneWithTiedParams();
		clone.linear1 = (LinearLayer) linear1.cloneWithTiedParams();
		clone.tanh = (TanhLayer) tanh.cloneWithTiedParams();
		clone.sentimentLinear2 = (LinearLayer) sentimentLinear2.cloneWithTiedParams();
		
		clone.lookup.link(clone.linear1);
		clone.linear1.link(clone.tanh);
		clone.tanh.link(clone.sentimentLinear2);
		
		clone.input = clone.lookup.input;
		
		clone.contextLinear2 = (LinearLayer) contextLinear2.cloneWithTiedParams();
		
		return clone;
	}
}
