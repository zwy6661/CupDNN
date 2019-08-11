package cpudnn.layer;



import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Vector;

import cpudnn.Network.Network;
import cpudnn.active.ReluActivationFunc;
import cpudnn.active.SigmodActivationFunc;
import cpudnn.active.TanhActivationFunc;
import cpudnn.data.Blob;
import cpudnn.util.MathFunctions;
import cpudnn.util.Task;
import cpudnn.util.ThreadPoolManager;

/*
 * ��ȿɷ�����
 * ��ȿɷ�����Ҫ�����ͨ��������ͨ����������
 * ��������ͨ����6�������ͨ��ֻ����6��12��18��24......
 */
public class DeepWiseConv2dLayer extends Layer{
	public static final String TYPE = "DeepWiseConv2dLayer";
	private Blob kernel;
	private Blob bias;
	private Blob kernelGradient;
	private Blob biasGradient;
	private Blob z;
	private int width;
	private int height;
	private int inChannel;
	private int outChannel;
	private int kernelSize;
	private int stride;
	
	public DeepWiseConv2dLayer(Network network){
		super(network);
	}

	public DeepWiseConv2dLayer(Network network, int width,int height,int inChannel,int outChannel,int kernelSize,int stride) {
		// TODO Auto-generated constructor stub
		super(network);
		this.width = width;
		this.height = height;
		this.inChannel = inChannel;
		this.outChannel = outChannel;
		this.kernelSize = kernelSize;
		this.stride = stride;
	}

	@Override
	public String getType() {
		// TODO Auto-generated method stub
		return TYPE;
	}

	@Override
	public void prepare() {
		// TODO Auto-generated method stub
		//layerParams.getHeight()��ʾ�ò���Ҫ��ȡ����������
		if(kernel ==null && bias == null){
			kernel = new Blob(outChannel,kernelSize,kernelSize);
			bias = new Blob(outChannel);
			//init params
			MathFunctions.gaussianInitData(kernel.getData());
			MathFunctions.constantInitData(bias.getData(), 0.001f);
		}
		z = new Blob(mNetwork.getBatch(),outChannel,height,width);
		kernelGradient = new Blob(outChannel,kernelSize,kernelSize);
		biasGradient = new Blob(outChannel);

	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		Blob input = mNetwork.getDatas().get(id-1);
		Blob output = mNetwork.getDatas().get(id);
		float [] outputData = output.getData();
		float [] zData = z.getData();
		
		//�����
		if(activationFunc!=null){
			//�����Ľ��������z��
			z.fillValue(0);
			MathFunctions.deepWiseConv2dSame(mNetwork,input, kernel, bias, z);
			Vector<Task<Object>> workers = new Vector<Task<Object>>();
			for(int n=0;n<output.getNumbers();n++){
				workers.add(new Task<Object>(n) {
					@Override
				    public Object call() throws Exception {
						for(int c=0;c<output.getChannels();c++){
							for(int h=0;h<output.getHeight();h++){
								for(int w=0;w<output.getWidth();w++){
									outputData[output.getIndexByParams(n, c, h, w)] = activationFunc.active(zData[z.getIndexByParams(n, c, h, w)]);
								}
							}
						}
						return null;
					}
				});
			}
			ThreadPoolManager.getInstance(mNetwork).dispatchTask(workers);
		}else {
			//�����Ľ��������output��
			output.fillValue(0);
			MathFunctions.deepWiseConv2dSame(mNetwork,input, kernel, bias, output);
		}
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		Blob input = mNetwork.getDatas().get(id-1);
		Blob inputDiff = mNetwork.getDiffs().get(id);
		Blob outputDiff = mNetwork.getDiffs().get(id-1);
		float[] inputDiffData = inputDiff.getData();
		float[] zData = z.getData();
		float[] kernelGradientData = kernelGradient.getData();
		float[] inputData = input.getData();
		float[] biasGradientData = biasGradient.getData();
		
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		if(activationFunc!=null){
			//�ȳ˼�����ĵ���,�õ��ò�����
			for(int n=0;n<inputDiff.getNumbers();n++){
				workers.add(new Task<Object>(n) {
					@Override
				    public Object call() throws Exception {
						for(int c=0;c<inputDiff.getChannels();c++){
							for(int h=0;h<inputDiff.getHeight();h++){
								for(int w=0;w<inputDiff.getWidth();w++){
									inputDiffData[inputDiff.getIndexByParams(n, c, h, w)] *= activationFunc.diffActive(zData[z.getIndexByParams(n, c, h, w)]);
								}
							}
						}
						return null;
					}
				});
			}
			ThreadPoolManager.getInstance(mNetwork).dispatchTask(workers);
		}
		
		//Ȼ����²���
		//����kernelGradient,���ﲢ������kernel,kernel���Ż����и���
		kernelGradient.fillValue(0);
		workers.clear();
		for(int n=0;n<inputDiff.getNumbers();n++){
			workers.add(new Task<Object>(n) {
				@Override
			    public Object call() throws Exception {
					for(int c=0;c<inputDiff.getChannels();c++){
						int inputChannelIndex = c/(inputDiff.getChannels()/input.getChannels());
						for(int h=0;h<inputDiff.getHeight();h++){
							for(int w=0;w<inputDiff.getWidth();w++){
								//�ȶ�λ�������λ��
								//Ȼ�����kernel,ͨ��kernel��λ�����λ��
								//Ȼ���������diff
								int inStartX = w - kernelGradient.getWidth()/2;
								int inStartY = h - kernelGradient.getHeight()/2;
								//�;���˳˼�
					
								for(int kh=0;kh<kernelGradient.getHeight();kh++){
									for(int kw=0;kw<kernelGradient.getWidth();kw++){
										int inY = inStartY + kh;
										int inX = inStartX + kw;
										if (inY >= 0 && inY < input.getHeight() && inX >= 0 && inX < input.getWidth()){
											kernelGradientData[kernelGradient.getIndexByParams(0, c, kh, kw)] += inputData[input.getIndexByParams(n,inputChannelIndex , inY, inX)]
													*inputDiffData[inputDiff.getIndexByParams(n, c, h, w)];
										}
									}
								}
							}
						}
					}
					return null;
				}
			});
		}
		ThreadPoolManager.getInstance(mNetwork).dispatchTask(workers);
		//ƽ��
		MathFunctions.dataDivConstant(kernelGradientData, inputDiff.getNumbers());
		
		//����bias
		biasGradient.fillValue(0);
		for(int n=0;n<inputDiff.getNumbers();n++){
			for(int c=0;c<inputDiff.getChannels();c++){
				for(int h=0;h<inputDiff.getHeight();h++){
					for(int w=0;w<inputDiff.getWidth();w++){
						biasGradientData[bias.getIndexByParams(0, 0, 0, c)] += inputDiffData[inputDiff.getIndexByParams(n, c, h, w)];
					}
				}
			}
		}
		//ƽ��
		MathFunctions.dataDivConstant(biasGradientData, inputDiff.getNumbers());
		
		if(id<=1)return;
		//�������ݶ��������
		outputDiff.fillValue(0);
		MathFunctions.deepWiseConv2dSame(mNetwork,inputDiff, kernel, outputDiff);	
		
		mNetwork.updateW(kernel, kernelGradient);
		mNetwork.updateW(bias, biasGradient);
	}

	@Override
	public void saveModel(ObjectOutputStream out) {
		// TODO Auto-generated method stub
		try {
			out.writeUTF(getType());
			out.writeUTF(getType());
			out.writeInt(width);
			out.writeInt(height);
			out.writeInt(inChannel);
			out.writeInt(outChannel);
			out.writeInt(kernelSize);
			out.writeInt(stride);
			out.writeObject(kernel);
			out.writeObject(bias);
			if(activationFunc != null){
				out.writeUTF(activationFunc.getType());
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	@Override
	public void loadModel(ObjectInputStream in) {
		// TODO Auto-generated method stub
		try {
			width = in.readInt();
			height = in.readInt();
			inChannel = in.readInt();
			outChannel = in.readInt();
			kernelSize = in.readInt();
			stride = in.readInt();
			kernel = (Blob) in.readObject();
			bias = (Blob) in.readObject();
			String activationType = in.readUTF();
			if(activationType.equals(ReluActivationFunc.TYPE)){
				setActivationFunc(new ReluActivationFunc());
			}else if(activationType.equals(SigmodActivationFunc.TYPE)){
				setActivationFunc(new SigmodActivationFunc());
			}else if(activationType.equals(TanhActivationFunc.TYPE)){
				setActivationFunc(new TanhActivationFunc());
			}
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public Blob createOutBlob() {
		// TODO Auto-generated method stub
		return new Blob(mNetwork.getBatch(),outChannel,height,width);
	}

	@Override
	public Blob createDiffBlob() {
		// TODO Auto-generated method stub
		return new Blob(mNetwork.getBatch(),outChannel,height,width);
	}
}
