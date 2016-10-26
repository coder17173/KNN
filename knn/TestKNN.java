package com.nust.ticket.knn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.csvreader.CsvReader;
import com.nust.ticket.entity.ticket;
import com.nust.ticket.utils.JCSVUtils;
import com.nust.ticket.utils.String2Vector;

public class TestKNN {
	public static final String PATH = "." + File.separator + "data"
			+ File.separator + "SGT(final).csv";
	public static final String PATH1 = "." + File.separator + "data"
			+ File.separator + "KNN" + File.separator + "3.txt";
	public static final String FEATUREPATH = "./data/feature.txt";

	/**
	 * 从数据文件中读取数据
	 * 
	 * @param datas
	 *            存储数据的集合对象
	 * @param path
	 *            数据文件的路径
	 * @throws IOException
	 */
	public void read(List<String[]> datas, String path) throws IOException {
		try {
			CsvReader reader = new CsvReader(path, ',', Charset.forName("SJIS"));
			while (reader.readRecord()) {
				datas.add(reader.getValues());
			}
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	/**
	 * 程序执行入口
	 * 
	 * @param args
	 * @throws IOException
	 */

	public static void main(String[] args) throws IOException {
		List<String> featureList = new ArrayList<String>();
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(FEATUREPATH))));
		String line = null;
		for (; (line = br.readLine()) != null;) {
			featureList.add(line.trim());
		}
		br.close();

		List<String[]> origDatas = JCSVUtils.readeCsv(PATH);
		List<String> ss = new ArrayList<String>();
		for (String[] tmp : origDatas) {
			ss.add(tmp[0]);
		}

		 double[][] vectors = String2Vector.binary(featureList, ss); // 将所有的summary转换为vector
		//double[][] vectors = String2Vector.TF(featureList, ss);
		 //double[][] vectors = String2Vector.DF(featureList, ss);
		 //double[][] vectors = String2Vector.IDF(featureList, ss);
		 //double[][] vectors = String2Vector.TFIDF(featureList, ss);
		 
		System.out.println("construct matrix finished");
		// ---------------------------------
/*		int len = featureList.size();
		double[] feature = new double[len];
		for (int i = 0; i < len; i++) {
			int count = 0;
			for (int j = 0; j < vectors.length; j++) {
				if (vectors[j][i] != 0) {
					count++;
				}
			}
			feature[i] = count;
		}*/
		
		// ---------------------------------

		List<ticket> datas = new ArrayList<ticket>();
		ticket t = null;
		for (int i = 0; i < origDatas.size(); i++) {
			t = new ticket(vectors[i]);
			t.FC1 = origDatas.get(i)[1];
			t.FC2 = origDatas.get(i)[2];
			t.id = origDatas.get(i)[3];
			datas.add(t);
		}

		
		// divided into ten groups
		List<List<ticket>> groups = new ArrayList<List<ticket>>();
		for (int i = 0; i < 10; i++) {
			List<ticket> temp = new ArrayList<ticket>();
			for (int j = i * 200; j < (i + 1) * 200; j++) {
				temp.add(datas.get(j));
			}
			groups.add(temp);
		}

		BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(PATH1))));

		/*double[] precisions = new double[10];
		double[] recalls = new double[10];
		double[] fs = new double[10];*/

//		double eprecision = 0.0, erecall = 0.0, ef1 = 0.0;

		for (int k = 1; k <= 15; k = k + 2) {
			double[] precisions = new double[10];
			double[] recalls = new double[10];
			double[] fs = new double[10];
			// 进行十重交叉验证
			for (int i = 0; i < 10; i++) {
				List<ticket> testDataList = groups.get(i);

				List<ticket> trainDataList = new ArrayList<ticket>();
				for (int j = 0; j < 10; j++) {
					if (j != i) {
						trainDataList.addAll(groups.get(j));
					}
				}
				
				
				//-----------------------
				int len = featureList.size();
				double[] feature = new double[len];
				for(int n = 0; n < len; n++)
				{
					double count = 0;
					for(int m = 0; m < trainDataList.size(); m++)
					{
						if(trainDataList.get(m).vector[n] != 0)
						{
							count++;
						}
					}
					feature[n] = count;
				}
				//-----------------------
				
				
				Map<String, Integer> categoryMap = new HashMap<String, Integer>();
				for(ticket tt : testDataList)
				{
					String FC2 = tt.FC2;
					if(categoryMap.containsKey(FC2))
					{
						categoryMap.put(FC2, categoryMap.get(FC2)+1);
					}else {
						categoryMap.put(FC2, 1);
					}
				}

				
				KNN knn = new KNN();
				int correctCount = 0, wrongCount = 0;
//				Map<String, List<Boolean>> recallComput = new HashMap<String, List<Boolean>>();
				Map<String, Integer> recallComput = new HashMap<String, Integer>();
				
				for (ticket t1 : testDataList) {
					double[] vector = t1.vector;
					// String category = knn.knn(trainDataList, vector, k);//K值变化
					//String category = knn.knn(trainDataList, vector, k, feature); // K值变化
					String category = knn.knn(trainDataList, vector, k, feature);
					boolean flag;
					if (category.trim().equals(t1.FC2.trim())) {
						flag = true;
						correctCount++;
					} else {
						flag = false;
						wrongCount++;
					}

					if (recallComput.containsKey(t1.FC2)) {
						// recallComput.get(s[2]).add(flag);
						/*List<Boolean> list = recallComput.get(t1.FC2);
						list.add(flag);
						recallComput.put(t1.FC2, list);*/
						if(flag == true){
							recallComput.put(t1.FC2, recallComput.get(t1.FC2)+1);
						}
					} else {
/*						List<Boolean> li = new ArrayList<Boolean>();
						li.add(flag);
						recallComput.put(t1.FC2, li);*/
						if(flag == true){
							recallComput.put(t1.FC2, 1);
						}
					}
					//System.out.println(category + "--------->" + t1.FC2 + "-------->" + flag);
					//bw.write(category + "--------->" + t1.FC2 + "-------->" + flag);
					//bw.newLine();
				}
				double precision = correctCount / (double) testDataList.size(); // precision
				double tempRecall = 0.0;
				for (String key : recallComput.keySet()) {
					/*double count = 0.0;
					List<Boolean> list = recallComput.get(key);
					for (int j = 0; j < list.size(); j++) {
						if (list.get(j) == true) {
							count++;
						}
					}*/
					double count = recallComput.get(key);
					
					tempRecall = tempRecall + count*1.0/categoryMap.get(key);
				}
				double recall = tempRecall / recallComput.size(); // recall
				double f1 = 2 * (precision * recall) / (precision + recall); // f1 score

				precisions[i] = precision;
				recalls[i] = recall;
				fs[i] = f1;
				//bw.write("---------------------" + i + "重结束------------------------------");
				//bw.newLine();
			}

			double sumprecision = 0.0, sumrecall = 0.0, sumf1 = 0.0;
			for (int i = 0; i < precisions.length; i++) {
				sumprecision += precisions[i];
				sumrecall += recalls[i];
				sumf1 += fs[i];
			}
			System.out.println("------------k = "+ k + "---------------------");
			System.out.println("正确率 = "+sumprecision/10+"    "+ "召回率为 = "+sumrecall/10+"      "+"f1 = "+sumf1/10);
			bw.write("------------k = "+ k + "---------------------");
			bw.newLine();
			bw.write("正确率 = "+sumprecision/10+"    "+ "召回率为 = "+sumrecall/10+"      "+"f1 = "+sumf1/10);
			bw.newLine();
			bw.write("---------------------------------------------------");
			bw.newLine();
		}
		bw.close();
	}
}
