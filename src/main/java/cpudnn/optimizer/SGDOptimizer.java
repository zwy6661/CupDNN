package cpudnn.optimizer;

import java.util.List;

import cpudnn.data.Blob;


/*
 * SGD without momentum
 */

public class SGDOptimizer extends Optimizer {
	
	public SGDOptimizer(float lr){
		super(lr);
	}

	
	public SGDOptimizer(float lr,Optimizer.GMode mode,float lamda){
		super(lr,mode,lamda);
	}

	@Override
	public void updateB(Blob b,Blob gradient) {
		// TODO Auto-generated method stub
		float[] bData = b.getData();
		float[] gradData = gradient.getData();
		for(int j=0;j<b.getSize();j++){
			bData[j] -= lr*gradData[j];
		}
	}
	@Override
	public void updateW(Blob w, Blob gradient) {
		// TODO Auto-generated method stub
			float[] wData = w.getData();
			float[] gradData = gradient.getData();
			if(mode==GMode.L2) {
				for(int j=0;j<w.getSize();j++){
					//���l2˥��
					wData[j] = (1.0f-lr*lamda)*wData[j]  - lr*gradData[j];
				}
			}else if(mode==GMode.L1){
				for(int j=0;j<w.getSize();j++){
					//���l1˥��
					if(wData[j]>=0) {
						wData[j] = wData[j] - lr*lamda - lr*gradData[j];
					}else {
						wData[j] = wData[j] + lr*lamda - lr*gradData[j];
					}
				}				
			}else {
				for(int j=0;j<w.getSize();j++){
					wData[j] -= lr*gradData[j];
				}
			}
	}
}
