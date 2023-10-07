using System;
namespace ValdeML
{
	public class LRS : iLRSF
	{
		public double Predict(Grad grad, double input)
		{
			return grad.w * input + grad.b;
		}
		//Experimental Optimizers.
        public double OptimizeB(Grad grad)
        {
			Bopt bop = grad.bop;
			double old_vdb = grad.b1 * bop.vdb + (1 - grad.b1) * grad.GetJB();
			bop.vdb = old_vdb;
			return bop.vdb;
        }
        public double OptimizeW(Grad grad)
        {
			Wopt wop = grad.wop;
			double old_wop = grad.b1 * wop.vdw + (1 - grad.b1) * grad.GetJW();
			wop.vdw = old_wop;
			return wop.vdw;
        }
		//Experimental Optimizers.
		public double[] InputDerivatives(Grad grad, double[] inputs)
		{
			double[] input_derivatives = new double[inputs.Length];
			for(int i = 0; i< inputs.Length; i++)
			{
				double input_derivative = grad.derivs[i] * inputs[i];
				input_derivatives[i] = input_derivative;
			}
			return input_derivatives;
		}
        public double[] ErrorDerivatives(Grad grad, double[] targets)
        {
			double[] error_derivatives = new double[targets.Length];
			for(int i= 0; i< targets.Length; i++)
			{
				double error_derivative = 2 * (grad.preds[i] - targets[i]) / targets.Length;
				error_derivatives[i] = error_derivative;
			}
			return error_derivatives;
        }
        public double[] Errors(Grad grad, double[] targets)
        {
			double[] errors = new double[targets.Length];
			for(int i = 0; i< targets.Length; i++)
			{
				double error = Math.Pow(grad.preds[i] - targets[i], 2) / (2 * targets.Length);
				errors[i] = error;
			}
			return errors;
        }
		public double[] Predictions(Grad grad, double[] inputs)
		{
			double[] predictions = new double[inputs.Length];
			for (int i = 0; i< inputs.Length; i++)
			{
				double prediction = grad.w * inputs[i] + grad.b;
				predictions[i] = prediction;
			}
			return predictions;
		}

        public void Train(Grad grad, SMODEL[][] batches)
		{
			while (grad.error >= 0)
			{
				if (grad.keep_training)
				{
					for(grad.bid= 0; grad.bid< batches.Length; grad.bid++)
					{
						SMODEL[] batch = batches[grad.bid];
						double[] inputs = batch.Select(x => x.input).ToArray();
						double[] targets = batch.Select(x => x.target).ToArray();

						grad.preds = Predictions(grad, inputs);
						grad.errors = Errors(grad, targets);
						grad.derivs = ErrorDerivatives(grad, targets);

						grad.input_derivs = InputDerivatives(grad, inputs);

						double tmp_w = grad.w - grad.a * grad.GetJW();
						//double tmp_w = grad.w - grad.a * OptimizeW(grad);
						grad.w = tmp_w;

						double tmp_b = grad.b - grad.a * grad.GetJB();
						//double tmp_b = grad.b - grad.a * OptimizeB(grad);
						grad.b = tmp_b;

						grad.GetError();

						if (grad.error <= Math.Pow(10, -4))
							break;
					}
					if (grad.error <= Math.Pow(10, -4))
						break;
				}
				else
					break;
			}
		}
    }
	public class LRM : iLRMF
    {
        public double Predict(Grad grad, double[] input)
        {
            double[] feature_calcs = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                feature_calcs[i] = grad.ws[i] * input[i];
            }
            return feature_calcs.Sum() + grad.b;
        }
        //Experimental Optimizers.
        public double OptimizeB(Grad grad)
		{
			Bopt bop = grad.bop;
			double old_vdb = grad.b1 * bop.vdb + (1 - grad.b1) * grad.GetJB();
			bop.vdb = old_vdb;
            return bop.vdb;
		}
		public double OptimizeW(Grad grad)
		{
			Wopt wop = grad.wops[grad.fid];
			double old_vdw = grad.b1 * wop.vdw + (1 - grad.b1) * grad.GetJW();
			wop.vdw = old_vdw;
            return wop.vdw;
		}
		//Experimental Optimizers.
        public double[] InputDerivatives(Grad grad, double[] inputs)
        {
			double[] input_derivatives = new double[inputs.Length];
			for(int i = 0; i< inputs.Length; i++)
			{
				double input_derivative = grad.derivs[i] * inputs[i];
				input_derivatives[i] = input_derivative;
			}
			return input_derivatives;
        }
        public double[] ErrorDerivatives(Grad grad, double[] targets)
        {
			double[] error_derivatives = new double[targets.Length];
			for(int i= 0; i< targets.Length; i++)
			{
				double error_derivative = (2 * (grad.preds[i] - targets[i])) / targets.Length;
				error_derivatives[i] = error_derivative;
			}
			return error_derivatives;
        }
        public double[] Errors(Grad grad, double[] targets)
        {
			double[] errors = new double[targets.Length];
			for(int i= 0; i< targets.Length; i++)
			{
				double error = Math.Pow(grad.preds[i] - targets[i], 2) / (2 * targets.Length);
				errors[i] = error;
			}
			return errors;
        }
        public double[] Predictions(Grad grad, double[][] inputs)
		{
			double[] predictions = new double[inputs.Length];
			for(int i= 0; i< inputs.Length; i++)
			{
				double[] input = inputs[i];
				double[] feats_calculations = new double[input.Length];
				for(int j= 0; j< input.Length; j++)
				{
					feats_calculations[j] = grad.ws[j] * input[j];
				}
				predictions[i] = feats_calculations.Sum() + grad.b;
			}
			return predictions;
		}
		
		public void Train(Grad grad, MMODEL[][] batches)
		{
			Transposer transposer = new Transposer();
			while(grad.error >= 0)
			{
				grad.epoch++;
				if(grad.keep_training)
				{ 
					for(grad.bid=0; grad.bid< batches.Length; grad.bid++)
					{
						MMODEL[] batch = batches[grad.bid];
						double[][] inputs = batch.Select(x => x.input).ToArray();
						double[] targets = batch.Select(x => x.target).ToArray();
						grad.d = batch.Length;

						grad.preds = Predictions(grad, inputs);
						grad.errors = Errors(grad, targets);
						grad.derivs = ErrorDerivatives(grad, targets);
						double[][] inputsT = transposer.TransposeList(inputs);
						
						for(grad.fid= 0; grad.fid< inputsT.Length; grad.fid++)
						{
							grad.input_derivs = InputDerivatives(grad, inputsT[grad.fid]);
							//double tmp_w = grad.ws[grad.fid] - (grad.a * grad.GetJW());
							double tmp_w = grad.ws[grad.fid] - grad.a * OptimizeW(grad);
							grad.ws[grad.fid] = tmp_w;
						}

						//double tmp_b = grad.b - (grad.a * grad.GetJB());
						double tmp_b = grad.b - grad.a * OptimizeB(grad);
                        grad.b = tmp_b;

						grad.GetError();
						if(grad.error <= Math.Pow(10, -4))
						{
							break;
						}
					}
                    if (grad.error <= Math.Pow(10, -4))
                    {
                        break;
                    }
                }
				else
				{
					break;
				}
			}
		}
	}
}

