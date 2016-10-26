package com.nust.ticket.knn;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

import com.nust.ticket.entity.ticket;
import com.nust.ticket.similarity.bow.Dice;
import com.nust.ticket.similarity.bow.EJaccard;
import com.nust.ticket.similarity.bow.Jaccard;
import com.nust.ticket.similarity.bow.NWordOverlap;
import com.nust.ticket.similarity.sequence.MCNLCSn;
import com.nust.ticket.similarity.sequence.NLCS;
import com.nust.ticket.similarity.sequence.SM;

public class KNN {
	/**
	 * 设置优先级队列的比较函数，距离越大，优先级越高
	 */
	private Comparator<KNNNode> comparator = new Comparator<KNNNode>() {
		public int compare(KNNNode o1, KNNNode o2) {
			if (o1.getDistance() >= o2.getDistance()) {
				return 1;
			} else {
				return 0;
			}
		}
	};

	/**
	 * 获取K个不同的随机数
	 * 
	 * @param k
	 *            随机数的个数
	 * @param max
	 *            随机数最大的范围
	 * @return 生成的随机数数组
	 */
	public List<Integer> getRandKNum(int k, int max) {
		List<Integer> rand = new ArrayList<Integer>(k);
		for (int i = 0; i < k; i++) {
			int temp = (int) (Math.random() * max);
			if (!rand.contains(temp)) {
				rand.add(temp);
			} else {
				i--;
			}
		}
		return rand;
	}


	/**
	 * sequence-based similarity
	 * @param s1
	 * @param s2
	 * @return
	 */
/*	public double calDistance(String s1, String s2) {
		double smSim = new SM().SMValue(s1, s2);
		double nlcsSim = new NLCS().Nlcs(s1, s2);
		double mcnlcsSim = new MCNLCSn().Mclcsn(s1, s2);
		double sim = (smSim+nlcsSim+mcnlcsSim)/3;
		return 1-sim;
	}*/
	
	/**
	 * bag-of-word similarity
	 * @param s1
	 * @param s2
	 * @return
	 */
	public double calDistance(String s1, String s2) {
		double jaccard = new Jaccard().jaccardDistance(s1, s2);
		double eJaccard = new EJaccard().EJaccardDis(s1, s2);
		double dice = new Dice().diceSim(s1, s2);
		double nWord = new NWordOverlap().overlapPhrase(s1, s2);
		double sim = (jaccard+eJaccard+dice+nWord)/4;
		return 1-sim;
	}

	/**
	 * Euclidean similarity
	 * @param d1
	 * @param d2
	 * @return
	 */
	public double EuclideanDis(double[] d1, double[] d2)
	{
		double sim = 0.0;
		for(int i = 0; i < d1.length; i++)
		{
			double dif = d1[i]-d2[i];
			sim += dif*dif;
		}
		return Math.sqrt(sim);
	}
	
	
	/**
	 * cosine similarity
	 * @param d1
	 * @param d2
	 * @return
	 */
	public double cosineDis(double[] d1, double[] d2)
	{
		double a = 0.0, b = 0.0, c = 0.0;
		for(int i = 0; i < d1.length; i++)
		{
			a += d1[i]*d2[i];
			b += d1[i]*d1[i];
			c += d2[i]*d2[i];
		}
		return 1-a/(Math.sqrt(b)*Math.sqrt(c));
	}
	
	
	/**
	 * Pairwise-adaptive [PAIR] similarity
	 * @param d1
	 * @param d2
	 * @return
	 */
	public double PAIRDis(double[] d1, double[] d2)
	{
		double a = 0.0, b = 0.0, c = 0.0;
		for(int i = 0; i < d1.length; i++)
		{
			if(d1[i] != 0 || d2[i] != 0)
			{
				a += d1[i]*d2[i];
				b += d1[i]*d1[i];
				c += d2[i]*d2[i];
			}
		}
		return 1-a/(Math.sqrt(b)*Math.sqrt(c));
	}

	
	/**
	 * Information-theoretic based document similarity
	 * @param d1
	 * @param d2
	 * @param feature
	 * @return
	 */
	public double ITDistance(double[] d1, double[] d2, double[] feature)
	{
		double a = 0.0, b = 0.0, c = 0.0;
		for(int i = 0; i < d1.length; i++)
		{
			double pi = feature[i];
			double min = (d1[i]<d2[i]?d1[i]:d2[i]);
			a += min*Math.log(pi);
			b += d1[i]*Math.log(pi);
			c += d2[i]*Math.log(pi);
		}
		return 1-(2*a/(b+c));
	}
	
	
	/**
	 * Similarity Measure for Text Processing
	 * @param d1
	 * @param d2
	 * @param f
	 * @return
	 */
	public double SMTPDis(double[] d1, double[] d2, double[] f)
	{
		double lamta = 0.5;
		double len = d1.length;
		double a = 0.0, b = 0.0;
		for(int i = 0; i < len; i++)
		{
			if(d1[i] != 0 && d2[i] != 0)
			{
				a += 0.5*(1 + Math.exp((d1[i]-d2[i])/f[i] * (d2[i]-d1[i])/f[i]));
//				a += 0.5*(1 + Math.exp(-Math.pow((d1[i]-d2[i])/f[i], 2)));
			}else if(d1[i] == 0 && d2[i] == 0){
				a += 0;
			}else{
				a += lamta;   //此处值变化
			}
			
			if(d1[i] == 0 && d2[i] == 0)
			{
				b += 0;
			}else{
				b += 1;
			}
		}
		
		double sim = (a/b + lamta)/(1 + lamta);
		return 1-sim;
	}
	
	
	/**
	 * 执行KNN算法，获取测试元组的类别
	 * @param datas   训练数据集
	 * @param testData   测试元组
	 * @param k    设定的K值
	 * @return 测试元组的类别
	 */
/*	public String knn(List<String[]> datas, String testData, int k) {
		PriorityQueue<KNNNode> pq = new PriorityQueue<KNNNode>(k, comparator);
		List<Integer> randNum = getRandKNum(k, datas.size()); // 产生K个随机数
		
		for (int i = 0; i < k; i++) {
			int index = randNum.get(i);
			String[] currData = datas.get(index); // 得到第i条
			String c = currData[2].toString(); // FC2
			KNNNode node = new KNNNode(index, calDistance(testData, currData[0]), c);
			pq.add(node);
		}
		
		for (int i = 0; i < datas.size(); i++) {
			String[] t = datas.get(i);
			double distance = calDistance(testData, t[0]); // 计算测试数据与每条记录之间的三种距离加权值
			KNNNode top = pq.peek();
			if (top.getDistance() > distance) {
				pq.remove();
				pq.add(new KNNNode(i, distance, t[2].toString()));
			}
		}

		return getMostClass(pq);
	}*/
	
	
	/**
	 * knn for vector space model
	 * @param datas
	 * @param testData
	 * @param k
	 * @return
	 */
	public String knn(List<ticket> datas, double[] testData, int k, double[] feature)
	{
		
		PriorityQueue<KNNNode> pq = new PriorityQueue<KNNNode>(k,comparator);
		List<Integer> randNum = getRandKNum(k, datas.size()); // 产生K个随机数
		for (int i = 0; i < k; i++) {
			int index = randNum.get(i);
			ticket currData = datas.get(index); // 得到第i条
			String c = currData.FC2;
			KNNNode node = new KNNNode(index, cosineDis(testData, currData.vector), c);
//			KNNNode node = new KNNNode(index, ITDistance(testData, currData.vector, feature), c);
//			KNNNode node = new KNNNode(index, SMTPDis(testData, currData.vector, feature), c);
			pq.add(node);
		}
		
		for (int i = 0; i < datas.size(); i++) {
			ticket t = datas.get(i);
			double distance = cosineDis(testData, t.vector); // 计算测试数据与每条记录之间的三种距离加权值
//			double distance = ITDistance(testData, t.vector, feature);
//			double distance = SMTPDis(testData, t.vector, feature);
			KNNNode top = pq.peek();
			if (top.getDistance() > distance) {
				pq.remove();
				pq.add(new KNNNode(i, distance, t.FC2));
			}
		}

		return getMostClass(pq);
	}
	

	/**
	 * 获取所得到的k个最近邻元组的多数类
	 * @param pq  存储k个最近近邻元组的优先级队列
	 * @return 多数类的名称
	 */
	private String getMostClass(PriorityQueue<KNNNode> pq) {
		Map<String, Integer> classCount = new HashMap<String, Integer>();
		for (int i = 0; i < pq.size(); i++) {
			KNNNode node = pq.remove();
			String c = node.getC();
			if (classCount.containsKey(c)) {
				classCount.put(c, classCount.get(c) + 1);
			} else {
				classCount.put(c, 1);
			}
		}
		int maxIndex = -1;
		int maxCount = 0;
		Object[] classes = classCount.keySet().toArray();
		for (int i = 0; i < classes.length; i++) {
			if (classCount.get(classes[i]) > maxCount) {
				maxIndex = i;
				maxCount = classCount.get(classes[i]);
			}
		}
		return classes[maxIndex].toString();
	}
}
