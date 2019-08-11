package cpudnn.cifar10;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import cpudnn.Network.Network;
import cpudnn.active.ReluActivationFunc;
import cpudnn.layer.Conv2dLayer;
import cpudnn.layer.FullConnectionLayer;
import cpudnn.layer.InputLayer;
import cpudnn.layer.PoolMaxLayer;
import cpudnn.layer.PoolMeanLayer;
import cpudnn.layer.SoftMaxLayer;
import cpudnn.loss.MSELoss;
import cpudnn.optimizer.SGDOptimizer;
import cpudnn.util.DigitImage;

public class Cifar10Network {
	Network network;
	SGDOptimizer optimizer;
	private void buildFcNetwork(){
		//��network��������
		InputLayer layer1 = new InputLayer(network,32,32,3);
		network.addLayer(layer1);
		FullConnectionLayer layer2 = new FullConnectionLayer(network,32*32*3,300);
		layer2.setActivationFunc(new ReluActivationFunc());
		network.addLayer(layer2);
		FullConnectionLayer layer3 = new FullConnectionLayer(network,300,30);
		layer3.setActivationFunc(new ReluActivationFunc());
		network.addLayer(layer3);
		FullConnectionLayer layer4 = new FullConnectionLayer(network,30,10);
		layer4.setActivationFunc(new ReluActivationFunc());
		network.addLayer(layer4);
		SoftMaxLayer sflayer = new SoftMaxLayer(network,10);
		network.addLayer(sflayer);
	}
	
	private void buildConvNetwork(){
		InputLayer layer1 = new InputLayer(network,32,32,3);
		network.addLayer(layer1);
		
		Conv2dLayer conv1 = new Conv2dLayer(network,32,32,3,6,5,1);
		conv1.setActivationFunc(new ReluActivationFunc());
		network.addLayer(conv1);
		
		PoolMaxLayer pool1 = new PoolMaxLayer(network,32,32,6,2,2);
		network.addLayer(pool1);
		
		Conv2dLayer conv2 = new Conv2dLayer(network,16,16,6,6,3,1);
		conv2.setActivationFunc(new ReluActivationFunc());
		network.addLayer(conv2);
		
		PoolMeanLayer pool2 = new PoolMeanLayer(network,16,16,6,2,2);
		network.addLayer(pool2);
		
		FullConnectionLayer fc1 = new FullConnectionLayer(network,8*8*6,256);
		fc1.setActivationFunc(new ReluActivationFunc());
		network.addLayer(fc1);
		
		FullConnectionLayer fc2 = new FullConnectionLayer(network,256,10);
		fc2.setActivationFunc(new ReluActivationFunc());
		network.addLayer(fc2);
		
		SoftMaxLayer sflayer = new SoftMaxLayer(network,10);
		network.addLayer(sflayer);
		
	}
	public void buildNetwork(int numOfTrainData){
		//���ȹ�����������󣬲����ò���
		network = new Network();
		network.setThreadNum(8);
		network.setBatch(20);
		//network.setLoss(new LogLikeHoodLoss());
		//network.setLoss(new CrossEntropyLoss());
		network.setLoss(new MSELoss());
		optimizer = new SGDOptimizer(0.1f);
		network.setOptimizer(optimizer);
		
		//buildFcNetwork();
		buildConvNetwork();

		network.prepare();
	}
	
	public void train(List<DigitImage> trainLists,int epoes,List<DigitImage> testLists) {
		network.train(trainLists, epoes, testLists);
	}
	
	public void test(List<DigitImage> imgList) {
		network.test(imgList);
	}
	
	public void saveModel(String name){
		network.saveModel(name);
	}
	
	public void loadModel(String name){
		network = new Network();
		network.loadModel(name);
		network.prepare();
	}
}
