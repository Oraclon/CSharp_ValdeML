using System;
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

	public class Model
	{
		public Node node1 = new Node("tanh");
		public Node node2 = new Node("tanh");
		public Node node3 = new Node("sigmoid");

		//Train Controls
		public int epochs { get; set; }
		public int epoch { get; set; }
		public int bid { get; set; }
		public int fid { get; set; }
		public bool keep_training = true;
		public int tolerance;
		//Error Variables.
		public double old_error { get; set; }
		public double[] errors { get; set; }
		public double[] error_derivs { get; set; }
		public double error = 0;
		//Model General Variables.
		public double a { get; set; }
		public int d { get; set; }
		public double e { get; set; }
		public double b1 = 0.9;
		public double b2 = 0.999;
		// Functions
		void UpdateError(string type)
		{
			double err = errors.Sum();
			if(type == "lls")
			{
				error = err / d;
			}
			else if (type == "mean")
			{
				error = err / (2 * d);
			}
		}
		public void SetLearningRate(double learning)
		{
			a = learning;
		}
		public void GetErrors(string type, double[] predictions, double[] targets)
		{
			int size     = targets.Length;
			errors       = new double[size];
			error_derivs = new double[size];
			double error = 0.0;
			double error_deriv = 0.0;
			for (int i = 0; i < size; i++)
			{
				if (type == "lls")
				{
					error       = targets[i] == 1 ? -Math.Log(predictions[i]) : -Math.Log(1 - predictions[i]);
					error_deriv = targets[i] == 1 ? -1 / predictions[i] : 1 / (1 - predictions[i]);
				}
				else if(type == "mean")
				{
					error       = Math.Pow(predictions[i] - targets[i], 2);
					error_deriv = 2 * (predictions[i] - targets[i]);
				}
				errors[i]       = error;
				error_derivs[i] = error_deriv;
			}
			UpdateError(type);
		}
	}

	public class Node
	{
		public Node(string activation_type)
		{
			type = activation_type;
		}
		//Slope Variables.
		public double w   = new Random().NextDouble() - 0.5;
        public double vdw = 0;
		public double sdw = 0;
		//Bias Variables.
		public double b   = 0;
		public double vdb = 0;
		public double sdb = 0;
		//Node Predictions and Activations.
		public string act_calc { get; set; }
		public string act_der_calc { get; set; }
		public double[] predictions { get; set; }
		public double[] activations { get; set; }
        //Node Derivatives.
        public double[] activation_derivatives { get; set; }
		public double[] jders { get; set; }
		public double[] jders_pows { get; set; }
		public double[] jwders { get; set; }
		public double[] jwders_pows { get; set; }
        //Prediction - Activation Control
        public string type { get; set; }

		public void Predict(double[] inputs)
		{
			int size = inputs.Length;
			predictions            = new double[size];
			activations            = new double[size];
			activation_derivatives = new double[size];
			for (int i = 0; i < size; i++)
			{
				double prediction            = w * inputs[i] + b;
				double activation            = 0.0;
				double activation_derivative = 0.0;

				if (type.Equals("tanh"))
				{
					act_calc              = "Math.Tanh(prediction)";
					act_der_calc          = "1-pow(activation, 2)";
                    activation            = Math.Tanh(prediction);
                    activation_derivative = 1 - Math.Pow(activation, 2);
                }
				else if(type.Equals("sigmoid"))
				{
					act_calc              = "1 / (1 + Math.Exp(-prediction))";
					act_der_calc          = "act*(1 - act)";
					activation            = 1 / (1 + Math.Exp(-1.0*prediction));
					activation_derivative = prediction * (1 - prediction);
				}
				//ReLU to be added.
				predictions[i]            = prediction;
				activations[i]            = activation;
                activation_derivatives[i] = activation_derivative;
			}
		}
		public void GetDerivs(double[] prev_deriv, double[] respect_to_inputs)
		{
			int size = respect_to_inputs.Length;
			jders       = new double[size];
			jders_pows  = new double[size];
			jwders      = new double[size];
			jwders_pows = new double[size];

			for (int i = 0; i < size; i++)
			{
				double jder      = prev_deriv[i] * activation_derivatives[i];
				double jder_pow  = Math.Pow(prev_deriv[i] * activation_derivatives[i], 2);
				jders[i]         = jder;
				jders_pows[i]    = jder_pow;
			}

			for (int i = 0; i < size; i++)
			{
				double jwder     = jders[i] * respect_to_inputs[i];
				double jwder_pow = Math.Pow(jders[i] * respect_to_inputs[i],2);
				jwders[i]        = jwder;
				jwders_pows[i]   = jwder_pow;
			}
		}
		public void UpdateGradient(Model model)
		{
            double tmp_w = w - model.a * jwders.Sum() / model.d;
            w = tmp_w;

            double tmp_b = b - model.a * jders.Sum() / model.d;
            b = tmp_b;
		}
	}
}