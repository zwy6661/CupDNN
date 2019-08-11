package cpudnn.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cpudnn.Network.Network;
import cpudnn.active.ActivationFunc;
import cpudnn.data.Blob;

public abstract class Layer{
	protected int id;
	protected Network mNetwork;
	protected ActivationFunc activationFunc;
	//BlobParams�е��ĸ�����˵��
	//��һ����batch,����һ���������ж��ٸ�ͼƬ
	//�ڶ�����channel,һ��ͼƬ�ж��ٸ�ͨ��
	//��������ͼƬ�ĸ�
	//���ĸ���ͼƬ�Ŀ�
	public Layer(Network network){
		this.mNetwork = network;
	}
	
	public void setId(int id){
		this.id = id;
	}
	public int getId(){
		return id;
	}
	abstract public Blob createOutBlob();
	abstract public Blob createDiffBlob();
	
	public void setActivationFunc(ActivationFunc func){
		this.activationFunc = func;
	}
	//����
	abstract public String getType();

	//׼������
	abstract public void prepare();
	
	//ǰ�򴫲��ͷ��򴫲�
	abstract public void forward();
	abstract public void backward();
	
	//���������װ��
	abstract public void saveModel(ObjectOutputStream out);
	abstract public void loadModel(ObjectInputStream in);
	
}