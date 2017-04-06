package funcs;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class Funcs {
	
//	public static DecimalFormat dFormat = new DecimalFormat("0.0000000000");
	
	public static void dumpEmbedFile(String embedFile, 
			String encoding,
			HashMap<String, Integer> vocabMap,
			double[][] table,
			int embeddingLength)
	{
		TreeMap<Integer, String> inverseVocabMap = new TreeMap<Integer, String>();
		for(String word: vocabMap.keySet())
		{
			inverseVocabMap.put(vocabMap.get(word), word);
		}
		
		try{
			PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(embedFile), encoding)));
			for(int idx: inverseVocabMap.keySet())
			{
				writer.write(inverseVocabMap.get(idx));
				for(int j = 0; j < embeddingLength; j++)
				{
					writer.write(" " + String.format("%.7f", table[idx][j]));
				}
				writer.write("\n");
			}
			writer.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public static HashMap<String, String> parseArgs(String[] args)
	{
		HashMap<String, String> argMap = new HashMap<String, String>();
		
		for(int i = 0; i < args.length; i++)
		{
			String key = args[i];
			if(key.startsWith("-"))
			{
				if(i + 1 < args.length)
				{
					argMap.put(args[i], args[i + 1]);
					i++;
				}
			}
		}
		return argMap;
	}
	
	public static void readTrainFile(
			String filePath, 
			String labelPath,
			String encoding, 
			List<Data> trainingDatas)
	{
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(filePath) , encoding));
			
			BufferedReader labelReader = new BufferedReader(new InputStreamReader(
					new FileInputStream(labelPath) , encoding));
			String line = null;
			String labelLine = null;
			while((line = reader.readLine()) != null && (labelLine = labelReader.readLine()) != null)
			{
				trainingDatas.add(new Data("<s> " + line + " </s>", 
						Integer.parseInt(labelLine.trim())) );
			}
			reader.close();
			labelReader.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public static void readTrainFile(
			String inPath, 
			String encoding, 
			int goldPol,
			List<Data> trainingDatas)
	{
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(inPath) , encoding));
			String line = null;
			while((line = reader.readLine()) != null)
			{
				trainingDatas.add(new Data("<s> " + line + " </s>", goldPol));
			}
			reader.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public static void minLengthSentence(List<String> trainFiles,
			String encoding)
	{
		int minLength = 999999;
		for(String fileP: trainFiles)
		{
			System.out.println("running " + fileP);
			try{
				BufferedReader reader = new BufferedReader(new InputStreamReader(
						new FileInputStream(fileP) , encoding));
				String line = null;
				while((line = reader.readLine()) != null)
				{
					if(line.split(" ").length < minLength)
					{
						minLength = line.split(" ").length;
					}
				}
				reader.close();
			}
			catch(IOException e){
				e.printStackTrace();
			}
		}
		
		System.out.println(minLength);
	}
	
	public static void getVocab(String vocabFile, 
			HashMap<String, Integer> vocabMap,
			String encoding)
	{
		try{
			BufferedReader reader = new BufferedReader(new InputStreamReader(
					new FileInputStream(vocabFile) , encoding));
			String line = null;
			while((line = reader.readLine()) != null)
			{
				String[] words = line.split(" ");
				String word = words[0];
				int idx = Integer.parseInt(words[1]);
				
				vocabMap.put(word, idx);
			}
			reader.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public static void getVocab(
			List<String> dataFiles,
			String encoding,
			HashMap<String, Integer> vocabMap,
			int miniFreq)
	{
		HashMap<String, Integer> wordFreqMap = new HashMap<String, Integer>();
		for(String inFile: dataFiles)
		{
			System.out.println("running " + inFile);
			try{
				BufferedReader reader = new BufferedReader(new InputStreamReader(
						new FileInputStream(inFile) , encoding));
				String line = null;
				while((line = reader.readLine()) != null)
				{
					String[] words = line.split(" ");
					for(String word: words)
					{
						if(!wordFreqMap.containsKey(word))
						{
							wordFreqMap.put(word, 0);
						}
						wordFreqMap.put(word, wordFreqMap.get(word) + 1);
					}
				}
				reader.close();
			}
			catch(IOException e){
				e.printStackTrace();
			}
		}
		
		System.out.println(wordFreqMap.size());
		TreeMap<Integer, List<String>> treeMap = new TreeMap<Integer, List<String>>();
		
		for(String word: wordFreqMap.keySet())
		{
			int freq = wordFreqMap.get(word);
			if(freq >= miniFreq)
			{
				if(!treeMap.containsKey(freq))
				{
					treeMap.put(freq, new ArrayList<String>());
				}
				treeMap.get(freq).add(word);
			}
		}
		
		vocabMap.put("<unk>", 0);
		vocabMap.put("<s>", 1);
		vocabMap.put("</s>", 2);
		int idx = 3;
		
		for(int freq: treeMap.descendingKeySet())
		{
			for(String word: treeMap.get(freq))
			{
				vocabMap.put(word, idx);
				idx++;
			}
		}
	}
	
	public static void dumpVocab(HashMap<String, Integer> hashMap, 
			String outputFile, 
			String encoding)
	{
		TreeMap<Integer, String> treeMap = new TreeMap<Integer, String>();
		for(String word: hashMap.keySet())
		{
			treeMap.put(hashMap.get(word), word);
		}
		
		try{
			PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
			          new FileOutputStream(outputFile), encoding)));
			for(int idx: treeMap.keySet())
			{
				writer.write(treeMap.get(idx) + " " + idx + "\n");
			}
			writer.close();
		}
		catch(IOException e){
			e.printStackTrace();
		}
	}
	
	public static int[] fillWindow(
			int beginIdx,
			Data data,
			int windowSize,
			HashMap<String, Integer> vocabMap) 
	{
		int[] wordIns = new int[windowSize];
		for(int i = 0; i < windowSize; i++)
		{
			String word = data.words[beginIdx + i];
			
			if(vocabMap.containsKey(word))
			{
				wordIns[i] = vocabMap.get(word);
			}
			else
			{
				wordIns[i] = vocabMap.get("<unk>");
			}
		}
		
		return wordIns;
	}
}
