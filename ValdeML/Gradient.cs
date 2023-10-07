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
		internal int d { get; set; }
		internal double e = Math.Pow(10, -8);
		//Training Vars
		internal double a = 0.4;
		internal int epoch = 0;
		internal double error = 0;
		internal bool keep_training = true;
		internal int bid { get; set; }
		internal int fid { get; set; }
		internal double[] preds { get; set; }
		internal double[] errors { get; set; }
		internal double[] derivs { get; set; }
		internal double[] input_derivs { get; set; }
		internal double[] pred_derivs { get; set; }
		//Dataset Vars
		internal SCALER scaler { get; set; }
		internal SCALER[] scalers { get; set; }
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
		internal void GetError()
		{
			error = errors.Sum();
		}
		internal double[] MScaleInput(double[] input)
		{
			double[] new_input = new double[input.Length];
			for(int i= 0; i< input.Length; i++)
			{
				SCALER scaler = scalers[i];
				if (scaler.type == "minmax")
					new_input[i] = (input[i] - scaler.min) / (scaler.max - scaler.min);
				else if (scaler.type == "mean")
					new_input[i] = (input[i] - scaler.m) / (scaler.max - scaler.min);
                else if (scaler.type == "zscore")
                    new_input[i] = (input[i] - scaler.m) / scaler.s;
            }
			return new_input;
		}
		internal double SScaleInput(double input)
		{
			double scaled= 0.0;
			if (scaler.type == "minmax")
				scaled = (input - scaler.min) / (scaler.max - scaler.min);
			else if (scaler.type == "mean")
				scaled = (input - scaler.m) / (scaler.max - scaler.min);
			else if (scaler.type == "zscore")
				scaled = (input - scaler.m) / scaler.s;
			else if (scaler.type == "maxsin")
				scaled = Math.Sin(((2 * Math.PI) * input) / scaler.max);
			else if (scaler.type == "maxcos")
                scaled = Math.Cos(((2 * Math.PI) * input) / scaler.max);

            return scaled;
		}
	}
}