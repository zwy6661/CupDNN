package cpudnn.util;

import java.util.Vector;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import cpudnn.Network.Network;

public class ThreadPoolManager {
	private static ThreadPoolManager instance;
	private static int threadNum = 4;
	public static ThreadPoolManager getInstance(Network network) {
		synchronized(ThreadPoolManager.class) {
			if(instance==null) {
				instance = new ThreadPoolManager(network.getThreadNum());
			}else if(network.getThreadNum()!=threadNum){
				//�߳�������������´���
				instance = new ThreadPoolManager(network.getThreadNum());
			}
		}
		return instance;
	}
	
	//1�������̳߳�
    private ThreadPoolExecutor threadPool;
	private ThreadPoolManager(int num) {
		if(num<1) {
			threadNum = 4;
			threadPool = new ThreadPoolExecutor(4, 4, 1000, TimeUnit.SECONDS, new LinkedBlockingDeque<Runnable>());
		}else {
			threadNum = num;
			threadPool = new ThreadPoolExecutor(num, num, 1000, TimeUnit.SECONDS, new LinkedBlockingDeque<Runnable>());
		}
	}
	public void dispatchTask(Vector<Task<Object>> workers) {
		//���ն��߳���Ӧ���
		Vector<Future<Object>> results = new Vector<Future<Object>>();
        for(Task<Object> c: workers) {
        	Future<Object> f = threadPool.submit(c);
        	results.add(f);
        }
        for(int i=0;i<results.size();i++) {
        	try {
				results.get(i).get();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
        }
	}
}
