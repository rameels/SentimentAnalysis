package sa_embedding;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

import funcs.Data;
import funcs.Funcs;

public class ContextRankingMain {
	public static void train(HashMap<String, String> argsMap) throws Exception
	{
		int xWindowSize = Integer.parseInt(argsMap.get("-windowSize"));
		int xHiddenSize = Integer.parseInt(argsMap.get("-hiddenLength"));
		int xEmbeddingLength = Integer.parseInt(argsMap.get("-embeddingLength"));
		String inputDir = argsMap.get("-inputDir");
		String vocabFile = argsMap.get("-vocabFile");
		int trainFileNum = Integer.parseInt(argsMap.get("-trainFileNum")); 
		// 99M corresponds to 99
		int trainRound = Integer.parseInt(argsMap.get("-trainingRound"));
		double learningRate = Double.parseDouble(argsMap.get("-learningRate"));
		double margin = Double.parseDouble(argsMap.get("-margin"));
		String outputFile = argsMap.get("-outputFile");
		double randomBase = Double.parseDouble(argsMap.get("-randomBase"));
		
		List<String> posFiles = new ArrayList<String>();
		List<String> negFiles = new ArrayList<String>();
		for(int i = 0; i < trainFileNum; i++)
		{
			posFiles.add(inputDir + "emoticon.pos." + i + ".txt");
			negFiles.add(inputDir + "emoticon.neg." + i + ".txt");
		}
		
		List<String> allTrainFiles = new ArrayList<String>();
		allTrainFiles.addAll(posFiles);
		allTrainFiles.addAll(negFiles);
		
		HashMap<String, Integer> vocabMap  = new HashMap<String, Integer>();
		
		Funcs.getVocab(vocabFile, vocabMap, "utf8");
		System.out.println("vocab.size(): " + vocabMap.size());
		
		ContextRanking posMain = new ContextRanking(
				xWindowSize, vocabMap.size(), xHiddenSize, xEmbeddingLength);
		
		Random rnd = new Random();
		posMain.randomize(rnd, -randomBase, randomBase);

		ContextRanking negMain = posMain.cloneWithTiedParams();
		
		double lossV = 0.0;
		int lossC = 0;
		for(int round = 0; round < trainRound; round++)
		{
			System.out.println("running round = " + round);
			
			Collections.shuffle(posFiles);
			Collections.shuffle(negFiles);
			
			for(int fileIdx = 0; fileIdx < posFiles.size(); fileIdx++)
			{
				List<Data> trainingDatas = new ArrayList<Data>();
				
				Funcs.readTrainFile(posFiles.get(fileIdx), "utf8", 
						0, trainingDatas);
				Funcs.readTrainFile(negFiles.get(fileIdx), "utf8", 
						1, trainingDatas);
				
				System.out.println("running pos-file: " + posFiles.get(fileIdx));
				System.out.println("running neg-file: " + negFiles.get(fileIdx));
				
				Collections.shuffle(trainingDatas);
				
				for(int dataIdx = 0; dataIdx < trainingDatas.size(); dataIdx++)
				{
					Data data = trainingDatas.get(dataIdx);
					if(data.words.length < xWindowSize)
					{
						continue;
					}
					
					for(int i = 0; i < data.words.length - xWindowSize + 1; i++)
					{
						int[] wordIns = Funcs.fillWindow(i, data, xWindowSize, vocabMap);
						System.arraycopy(wordIns, 0, posMain.input, 0, xWindowSize);
						System.arraycopy(wordIns, 0, negMain.input, 0, xWindowSize);

						int randWordIdx = rnd.nextInt(vocabMap.size());
						while(randWordIdx == wordIns[xWindowSize/2])
						{
							randWordIdx = rnd.nextInt(vocabMap.size());
						}
						negMain.input[xWindowSize/2] = randWordIdx;
						
						posMain.forward();
						negMain.forward();
						
						lossC += 1;
						// loss function
						if(posMain.output[0] > negMain.output[0] + margin)
						{
							continue;
						}
						
						lossV += margin + negMain.output[0] - posMain.output[0];
						
						posMain.outputG[0] = 1;
						negMain.outputG[0] = -1;
						
						posMain.backward();
						negMain.backward();
						
						posMain.update(learningRate);
						negMain.update(learningRate);
						
						posMain.clearGrad();
						negMain.clearGrad();
					}
					
					if(dataIdx % 50000 == 0)
					{
						System.out.println("running " + dataIdx + "/" + trainingDatas.size() + 
								"\t loss: " + (lossV / lossC) + "\t" + new Date().toLocaleString());
					}
				}
				
				trainingDatas.clear();
			}
			
			Funcs.dumpEmbedFile(outputFile + "-round-" + round,
					"utf8", vocabMap, posMain.lookup.table, xEmbeddingLength);
		}
	}
	
	public static void main(String[] args) {
		
		HashMap<String, String> argsMap = Funcs.parseArgs(args);
		for(String key: argsMap.keySet())
		{
			System.out.println(key + "\t" + argsMap.get(key));
		}
		try {
			train(argsMap);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

}
