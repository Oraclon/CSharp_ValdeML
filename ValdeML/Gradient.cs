using System;
namespace ValdeML
{
	public class Wopt
	{
		internal double vdw = 0;
		internal double sdw = 0;
	}
	public class Bopt
	{
		internal double vdb = 0;
		internal double sdb = 0;
	}
	public class Grad
	{
		//Gradient Vars
		internal double w = 0.6;
		internal double[] ws { get; set; }
		internal double b = 0;
		//Optimizer Vars
		internal Wopt wop = new Wopt();
		internal Bopt bop = new Bopt();
		internal Wopt[] wops { get; set; }
		internal Bopt[] bops { get; set; }
		internal double b1 = 0.9;
		internal double b2 = 0.999;
		//Training Vars
		internal double a = 0.4;
		internal int epoch = 0;
		internal int bid { get; set; }
		internal int fid { get; set; }
		internal double[] preds { get; set; }
		internal double[] errors { get; set; }
		internal double[] derivs { get; set; }
		internal double[] input_derivs { get; set; }
		//Dataset Vars
		internal SCALER holder { get; set; }
		internal SCALER[] holders { get; set; }
		//Gradient Voids
		internal double GetJW()
		{
			return input_derivs.Sum();
		}
		internal double GetJB()
		{
			return derivs.Sum();
		}
		internal void UpdateW(double[] input)
		{
			Random random = new Random();
			ws = new double[input.Length];
			wops = new Wopt[input.Length];
			for(int x= 0; x< input.Length; x++)
			{
				ws[x] = random.NextDouble() - .5;
				wops[x] = new Wopt();
			}
		}
	}
}