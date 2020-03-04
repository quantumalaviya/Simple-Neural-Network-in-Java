package utils;
import java.lang.Math;


public class util 
{
	
	public static double[][] matmul(double[][] A, double[][] B) throws ShapeException
	{
		if(A[0].length!=B.length)
			throw new ShapeException();
		
		double[][] C = new double[A.length][B[0].length];
		
		for(int i = 0; i < A.length; i++)
		{
			for(int j = 0; j < B[0].length; j++)
			{
				C[i][j] = 0;
				for(int k = 0; k < A[0].length; k++)
				{
					C[i][j] += (A[i][k] * B[k][j]);
				}
			}
		}
		return C;
	}
	
	public static double[][] dotmul(double[][] A, double[][] B) throws ShapeException
	{
		double[][] C = new double[B.length][B[0].length];
		
		
		if(A.length!=B.length||A[0].length!=B[0].length)
			throw new ShapeException();
		
		
		for(int i = 0; i < B.length; i++)
		{
			for(int j = 0; j < B[i].length; j++)
			{
				C[i][j] = A[i][j] * B[i][j];
			}
		}
		return C;
	}
	
	public static double[][] multiply(double[] A, double[][] B) throws ShapeException
	{
		double[][] C = new double[B.length][B[0].length];
		
		if(A.length!=B[0].length)
			throw new ShapeException();
		
		
		for(int i = 0; i < B.length; i++)
		{
			for(int j = 0; j < B[i].length; j++)
			{
				C[i][j] = A[j] * B[i][j];
			}
		}
		return C;
	}
	
	public static double[][] divide(double[] A, double[][] B) throws ShapeException
	{
		double[][] C = new double[B.length][B[0].length];
		
		if(A.length!=B[0].length)
			throw new ShapeException();
		
		
		for(int i = 0; i < B.length; i++)
		{
			for(int j = 0; j < B[i].length; j++)
			{
				C[i][j] = A[j] / B[i][j];
			}
		}
		return C;
	}
	
	public static double[][] matadd(double[][] A, double[] B) throws ShapeException
	{
		double[][] C = new double[A.length][A[0].length];
		
		if(A.length!=B.length)
			throw new ShapeException();
		
		
		for(int i = 0; i < A.length; i++)
		{
			for(int j = 0; j < A[i].length; j++)
			{
				C[i][j] = A[i][j] + B[i];
			}
		}
		return C;
	}

	public static double[][] matadd(double[][] A, double[][] B) throws ShapeException
	{
		if(A.length!=B.length||A[0].length!=B[0].length)
			throw new ShapeException();
		
		double[][] C = new double[A.length][A[0].length];
		
		for(int i = 0; i < A.length; i++)
		{
			for(int j = 0; j < A[i].length; j++)
			{
				C[i][j] = A[i][j] + B[i][j];
			}
		}
		return C;
	}
	
	public static double[][] sigmoid(double[][] A)
	{
		double[][] C = new double[A.length][A[0].length];
		for(int i = 0; i < A.length; i++)
		{
			for(int j = 0; j < A[i].length; j++)
			{
				C[i][j] = (double)Math.exp(-A[i][j]);
				C[i][j]+=1;
				C[i][j]=1/C[i][j];
			}
		}
		return C;
	}

	public static double[][] relu(double[][] A)
	{
		double[][] C = new double[A.length][A[0].length];
		for(int i = 0; i < A.length; i++)
		{
			for(int j = 0; j < A[i].length; j++)
			{
				C[i][j] = A[i][j]>0?A[i][j]:0;
			}
		}
		return C;
	}

	public static double[][] Transpose(double[][] A)
	{
		double[][] C = new double[A[0].length][A.length];
		for(int i = 0; i < C.length; i++)
		{
			for(int j = 0; j < C[i].length; j++)
			{
				C[i][j] = A[j][i];
			}
		}
		return C;
	}
	
	public static double[][] reluBackward(double[][] dA, double[][] Z) throws ShapeException
	{
		if(dA.length!=Z.length||dA[0].length!=Z[0].length)
			throw new ShapeException();
		
		double[][] dZ = new double[dA.length][dA[0].length];

		for(int i = 0; i < dA.length; i++)
		{
			for(int j = 0; j < dA[i].length; j++)
				dZ[i][j] = Z[i][j]<=0?0:dA[i][j];
		}
		
		return dZ;
	}
	
	public static double[][] sigmoidBackward(double[][] dA, double[][] Z) throws ShapeException
	{
		if(dA.length!=Z.length||dA[0].length!=Z[0].length)
			throw new ShapeException();
		double[][] dZ = new double[dA.length][dA[0].length];
		
		
		
		double s[][] = sigmoid(Z);


		for(int i = 0; i < s.length; i++)
		{
			for(int j = 0; j < s[i].length; j++)
				s[i][j]*=(1-s[i][j]);
		}

		dZ = dotmul(dA, s);
		
		return dZ;
	}
}