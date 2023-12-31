﻿using System;
namespace ValdeML
{
	public class Wopt
	{
		internal double w { get; set; }
		internal double vdw = 0;
		internal double sdw = 0;
	}
	public class Bopt
	{
		internal double b = 0;
		internal double vdb = 0;
		internal double sdb = 0;
	}
	public class Grad
	{
		//Gradient Vars
		internal Wopt w = new Wopt();
		internal Wopt[] ws { get; set; }
		internal Bopt b = new Bopt();
		//Optimizer Vars
		internal Wopt wop = new Wopt();
		internal Bopt bop = new Bopt();
		internal Wopt[] wops { get; set; }
		internal Bopt[] bops { get; set; }
		internal double b1 = 0.9;
		internal double b2 = 0.999;
		internal int d { get; set; }
		internal double e = Math.Pow(10, -8);
        //Control Vars
        internal double error = 0;
		internal double old_error = 0;
		internal double tolerance = 0;
		internal double tolerance_step = 0;
        internal bool keep_training = true;
        internal int bid { get; set; }
        internal int fid { get; set; }
        //Training Vars
        internal double a = 0.4;
		internal int epoch = 0;
		internal double[] preds { get; set; }
		internal double[] pred_activations { get; set; }
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
			int size = input_derivs.Length;
			return input_derivs.Sum() / size;
		}
		internal double GetJB()
		{
			int size = derivs.Length;
			return derivs.Sum() / size;
		}
		internal void UpdateW(double[] input)
		{
			Random random = new Random();
			int size = input.Length;
			ws = new Wopt[size];
			for (int i = 0; i < size; i++)
			{
				Wopt wop = new Wopt();
				wop.w = random.NextDouble() * Math.Pow(10, -1);
				ws[i] = wop;
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
		internal void SetTolerance(int var)
		{
			tolerance = var;
		}
		internal void CheckError()
		{
			if(!error.Equals(old_error))
			{
				old_error = error;
				string msg = $"{error} {old_error} {old_error - error}";
				Console.WriteLine(msg);
			}
			else
			{
                tolerance_step++;
				if(tolerance_step.Equals(tolerance))
				{
					keep_training = false;
				}
			}
		}
	}

	
}
