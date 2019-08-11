package cpudnn.mnist;

import java.io.IOException;
import java.util.List;

import cpudnn.util.DigitImage;




public class MnistTest {
	static List<DigitImage> trains = null ;
	static List<DigitImage> tests = null;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		//load mnist
		ReadFile rf1=new ReadFile("C:\\Users\\22682\\Desktop\\CupDnn-master\\data\\mnist\\train-labels.idx1-ubyte",
				"C:\\Users\\22682\\Desktop\\CupDnn-master\\data\\mnist\\train-images.idx3-ubyte");
		ReadFile rf2=new ReadFile("C:\\Users\\22682\\Desktop\\CupDnn-master\\data\\mnist\\t10k-labels.idx1-ubyte",
				"C:\\Users\\22682\\Desktop\\CupDnn-master\\data\\mnist\\t10k-images.idx3-ubyte");
		try {
			tests = rf2.loadDigitImages();
			trains =rf1.loadDigitImages();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		MnistNetwork mn = new MnistNetwork();
		mn.buildNetwork(trains.size());
		mn.train(trains,30,tests);

		mn.saveModel("C:\\Users\\22682\\Desktop\\CupDnn-master\\model\\mnist.model");
		
		
		mn.loadModel("C:\\Users\\22682\\Desktop\\CupDnn-master\\model\\mnist.model");
		mn.test(tests);

	}

}
