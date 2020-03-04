package Param;

public class Parameters
{

	public double valueW[][];
	public double valueb[];

	public void putParameters(double[][] a, double[] b)
	{
		valueW = a;
		valueb = b;	
	}

	public void dispParameters()
	{
		double[][] a = valueW;
		double[] b = valueb;
	
		for(int j = 0; j < a.length; j++)
		{
			for(int k = 0; k < a[j].length; k++)
			{
				System.out.println(a[j][k]);
			}
		}

		for(int k = 0; k < b.length; k++)
		{
			System.out.println(b[k]);
		}
	}
}
