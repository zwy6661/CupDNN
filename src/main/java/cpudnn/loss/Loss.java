package cpudnn.loss;

import cpudnn.data.Blob;

public abstract class Loss {
	abstract public float loss(Blob label,Blob output);
	abstract public void diff(Blob label,Blob output,Blob diff);
}
