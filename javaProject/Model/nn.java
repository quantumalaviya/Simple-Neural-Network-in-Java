package Model;
import java.util.*;
import java.lang.*;
import utils.*;;
import Param.Parameters;

class Cache
{
	double A[][], W[][], b[], Z[][];
	
	void putCache(double[][] x, double[][] y, double[] z)
	{
		A = x;
		W = y;
		b = z;
	}
	
	void putZ(double[][] x)
	{
		Z = x;
	}
	
}

class Pair
{
	double[][] A;
	Cache cache, cachef[];
	
	Pair(double[][] a, Cache c)
	{
			A = a;
			cache = c;
	}

	Pair(double[][] a, Cache[] c)
	{
			A = a;
			cachef = c;
	}
	
	double[][] getKey()
	{
		return A;
	}
	
	Cache getValue()
	{
		return cache;
	}
	
	Cache[] getValuef()
	{
		return cachef;
	}
}

class returnType
{
	double dA_prev[][], dW[][], db[];
	returnType(double[][] da_prev, double[][] dw, double[] dB)
	{
		dA_prev = da_prev;
		dW = dw;
		db = dB;
	}
}

public class nn
{

	public static Parameters[] initializeParameters(int[] layerDims)
	{
		int L = layerDims.length;
		Parameters[] parameters = new Parameters[L-1];
		
		for(int i=1; i<L; i++)
		{	
			parameters[i-1] = new Parameters();
			double a[][] = new double[layerDims[i]][layerDims[i-1]];
			double b[] = new double[layerDims[i]];

			for(int j = 0; j < a.length; j++)
			{
				for(int k = 0; k < a[j].length; k++)
				{
					a[j][k] = (double)(Math.random()*0.01);
				}
			}

			for(int k = 0; k < b.length; k++)
			{
				b[k] = 0;
			}

			parameters[i-1].putParameters(a,b);
		}	
	    return parameters;
	}

	public static Pair forward(double[][] A, double[][] W, double[] b)
	{

		Cache cache = new Cache();
		util obj = new util();
			
		double Z[][] = new double[W.length][A[0].length];
		try
		{	
			Z = obj.matadd(obj.matmul(W, A), b);
		}
		catch(ShapeException a){}	
			cache.putCache(A, W, b);
			
			return new Pair(Z, cache);
	}

	public static Pair activationForward(double[][] A_prev, double[][] W, double[] b, String activation)
	{
		Cache cache = new Cache();
		util obj = new util();
		
		double Z[][] = new double[W.length][A_prev[0].length];
		double A[][] = new double[W.length][A_prev[0].length];
		
		Pair x = forward(A_prev, W, b);
		Z = x.getKey();
		cache = x.getValue();		
		
		if (activation == "sigmoid")
			A = obj.sigmoid(Z);
		
		else if (activation == "relu")
			A = obj.relu(Z);
		
		cache.putZ(Z);
		
		return new Pair(A, cache);
	}

	public static Pair finalForward(double X[][], Parameters[] parameters)
	{
		int L = parameters.length;
		Cache caches[] = new Cache[L];
		double A[][] = X;
		int l;
		Pair x;
		
		for(l = 1; l < L; l++)
		{
			double A_prev[][] = A;
			x = activationForward(A_prev, parameters[l-1].valueW, parameters[l-1].valueb,"relu");
			A = x.getKey();
			caches[l-1] = x.getValue();
		}
		
		x = activationForward(A, parameters[l-1].valueW, parameters[l-1].valueb,"sigmoid");
		double AL[][] = x.getKey();
		caches[l-1] = x.getValue();
		
		return new Pair(AL, caches);
	}
	
	public static double computeCost(double[][] AL, double[] Y)
	{
		int m = Y.length;
		util obj = new util();
		double cost = 0;
		double sY[] = new double[m];
		double sAL[][] = new double[AL.length][AL[0].length];
		double lAL[][] = new double[AL.length][AL[0].length];
		
		for(int i = 0; i < Y.length; i++)
			sY[i] = (1-Y[i]);
		
		for(int i = 0; i < AL.length; i++)
		{
			for(int j = 0; j < AL[i].length; j++)
				sAL[i][j] = (double)Math.log(1-AL[i][j]);
		}
		
		for(int i = 0; i < AL.length; i++)
		{
			for(int j = 0; j < AL[i].length; j++)
				lAL[i][j] = (double)Math.log(AL[i][j]);
		}
		try
		{
				
			double[][] a = obj.multiply(Y, lAL);
			double[][] b = obj.multiply(sY, sAL);
			double[][] c = obj.matadd(a,b);

		
		for(int i = 0; i < c.length; i++)
		{
			cost = 0;
			for(int j = 0; j < c[0].length; j++)
				cost += c[i][j];
		}
		cost = -1*cost/m;
		}
		catch(ShapeException a){}
		return cost;

	}
	
	public static returnType backward(double[][] dZ, Cache cache)
	{
		util obj = new util();
		
		double A_prev[][] = cache.A;
		double W[][] = cache.W;
		double b[] = cache.b;
		
		double dW[][] = new double[W.length][W[0].length];
		double db[] = new double[b.length];
		double dA_prev[][] = new double[A_prev.length][A_prev[0].length];
		
		db = new double[dZ.length];
		
		int m = A_prev[0].length;
		try{
			dW = obj.matmul(dZ, obj.Transpose(A_prev));
		
			for(int i = 0; i < dW.length; i++)
			{
				for(int j = 0; j < dW[i].length; j++)
					dW[i][j]/=m;
			}
			
			for(int i = 0; i < dZ.length; i++)
			{
				db[i] = 0;
				for(int j = 0; j < dZ[i].length; j++)
					db[i] += dZ[i][j];
				db[i]/=m;
			}
			
			dA_prev = obj.matmul(obj.Transpose(W), dZ);
		}
		catch(ShapeException a){}	

		return new returnType(dA_prev, dW, db);
	}

	public static returnType activationBackward(double[][] dA, Cache cache, String activation)
	{
		util obj = new util();
		double dZ[][], dA_prev[][], dW[][], db[];
		
		dZ = new double[dA.length][dA[0].length];
		try{
				
			if (activation=="relu")
				dZ = obj.reluBackward(dA, cache.Z);
			
			else if (activation=="sigmoid")
				dZ = obj.sigmoidBackward(dA, cache.Z);
		
		}
		catch(ShapeException a){}
		
		returnType ret = backward(dZ, cache);
		
		return ret;
	}
	
	public static returnType[] finalBackward(double[][] AL, double[] Y, Cache[] caches)
	{
		util obj = new util();
		int L = caches.length;
		int m = Y.length;
		returnType[] grads = new returnType[L];
		Cache current = new Cache();
		
		
		double sY[] = new double[m];
		double sAL[][] = new double[AL.length][AL[0].length];
		double dAL[][];
		double a[][] = new double[AL.length][AL[0].length];
		double b[][] = new double[AL.length][AL[0].length];
		
		
		dAL = new double[AL.length][AL[0].length];
		
		for(int i = 0; i < Y.length; i++)
			sY[i] = 1-Y[i];
		
		for(int i = 0; i < dAL.length; i++)
		{
			for(int j = 0; j < dAL[i].length; j++)
				sAL[i][j] = 1-AL[i][j];
		}
		try
		{
			a = obj.divide(Y, AL);
			b = obj.divide(sY, sAL);
		}
		catch(ShapeException v){}
		
		for(int i = 0; i < dAL.length; i++)
		{
			for(int j = 0; j < dAL[i].length; j++)
			{
				dAL[i][j] = -a[i][j]+b[i][j];
			}
		}	
		
		current = caches[L-1];
		returnType ret = activationBackward(dAL, current, "sigmoid");
		grads[L-1] = ret;
		
		for(int l = L-2; l >= 0 ; l--)
		{
			current = caches[l];
			ret = activationBackward(grads[l+1].dA_prev, current, "relu");
			grads[l] = ret;
		}
		
		return grads;
	}
	
	public static Parameters[] update(Parameters[] parameters, returnType[] grads, double lr)
	{
		int L = parameters.length;
		for(int i = 0; i < L; i++)
		{
			for(int j = 0; j < parameters[i].valueW.length; j++)
			{
				for(int k = 0; k < parameters[i].valueW[j].length; k++)
				{
					parameters[i].valueW[j][k] = parameters[i].valueW[j][k] - lr*grads[i].dW[j][k];
				}
			}
			
			for(int k = 0; k < parameters[i].valueb.length; k++)
			{
				parameters[i].valueb[k] = parameters[i].valueb[k] - lr*grads[i].db[k];
			}
		}	
		return parameters;
	}

	public static Parameters[] fit(double[][] X, double[] Y, int[] layerDims, double lr, int iter)
	{
		int L = layerDims.length;
		double AL[][];
		Cache[] caches = new Cache[L-1];
		returnType[] grads = new returnType[L-1];
		
		Parameters[] parameters = initializeParameters(layerDims);

		System.out.println("Done initialize");
		
		for(int i = 0; i < iter; i++)
		{
			Pair p = finalForward(X, parameters);
			AL = p.A;
			caches = p.cachef;
						
			double cost = computeCost(AL, Y);
			
			grads = finalBackward(AL, Y, caches);
			
			parameters = update(parameters, grads, lr);
			
			if (i%100 == 0)
			{
				System.out.println("Cost after iteration " + i + ":" + cost);
			}	
		}
		return parameters;
	}
	
	public static double[] predict(double X[][], Parameters[] parameters)
	{
		int L = parameters.length;
		Cache caches[] = new Cache[L];
		double A[][] = X;
		int l;
		Pair x;
		
		for(l = 1; l < L; l++)
		{
			double A_prev[][] = A;
			x = activationForward(A_prev, parameters[l-1].valueW, parameters[l-1].valueb,"relu");
			A = x.getKey();
			caches[l-1] = x.getValue();
		}
		
		x = activationForward(A, parameters[l-1].valueW, parameters[l-1].valueb,"sigmoid");
		double AL[][] = x.getKey();
		caches[l-1] = x.getValue();
		
		return AL[0];
	}
	
}



