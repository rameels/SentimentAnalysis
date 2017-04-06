package funcs;

public class Data {

	public String text;
	public int goldPol;
	public String[] words;
	
	public Data(String xText, int xGoldPol)
	{
		text = xText;
		goldPol = xGoldPol;
		words = text.split(" ");
	}
}
