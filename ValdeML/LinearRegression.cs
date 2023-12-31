﻿using System;
using ValdeML;

namespace ValdeML
{
	public class LRS : iLRSF
	{
        //To be done.
        public double OptimizeW(Grad grad)
        {
            throw new NotImplementedException();
        }

        public double OptimizeB(Grad grad)
		{
			throw new NotImplementedException();
		}
        //To be done.

		public double Predict(Grad grad, double input)
		{
            return grad.w.w * input + grad.b.b;
		}

        public double[] InputDerivatives(Grad grad, double[] inputs)
        {
            int size = inputs.Length;
            double[] input_derivatives = new double[size];

            for (int i = 0; i < size; i++)
            {
                double input_derivative = grad.derivs[i] * inputs[i];
                input_derivatives[i] = input_derivative;
            }

            return input_derivatives;
        }

        public double[] ErrorDerivatives(Grad grad, double[] targets)
        {
            int size = targets.Length;
            double[] error_derivs = new double[size];

            for(int i = 0; i< size; i++)
            {
                double error_deriv = 2 * (grad.preds[i] - targets[i]);
                error_derivs[i] = error_deriv;
            }

            return error_derivs;
        }

        public double[] Errors(Grad grad, double[] targets)
        {
            int size = targets.Length;
            double[] errors = new double[size];

            for (int i = 0; i < size; i++)
            {
                double error = Math.Pow(grad.preds[i] - targets[i], 2);
                errors[i] = error;
            }

            grad.error = errors.Sum() / (2 * targets.Length);
            return errors;
        }

        public double[] Predictions(Grad grad, double[] inputs)
		{
            int size = inputs.Length;
            double[] predictions = new double[size];
            for (int i = 0; i < size; i++)
            {
                double prediction = grad.w.w * inputs[i] + grad.b.b;
                predictions[i] = prediction;
            }
            return predictions;
		}


        public void Train(Grad grad, SMODEL[][] batches, bool optim_activated)
		{
            while (grad.error >= 0)
            {
                grad.epoch++;
                for (grad.bid = 0; grad.bid < batches.Length; grad.bid++)
                {
                    SMODEL[] batch = batches[grad.bid];
                    double[] inputs = batch.Select(x => x.input).ToArray();
                    double[] targets = batch.Select(x => x.target).ToArray();
                    grad.d = batch.Length;

                    grad.preds = Predictions(grad, inputs);
                    grad.errors = Errors(grad, targets);
                    grad.derivs = ErrorDerivatives(grad, targets);
                    grad.input_derivs = InputDerivatives(grad, inputs);

                    Wopt wop = grad.w;
                    Bopt bop = grad.b;

                    double tmp_w = wop.w - grad.a * grad.GetJW();
                    wop.w = tmp_w;

                    double tmp_b = bop.b - grad.a * grad.GetJB();
                    bop.b = tmp_b;

                    if (grad.error <= Math.Pow(10, -2))
                        break;
                }
                if (grad.error <= Math.Pow(10, -2))
                    break;
            }
		}
	}

    public class LRM : iLRMF
    {
        //To be done.
        public double OptimizeW(Grad grad)
        {
            throw new NotImplementedException();
        }

        public double OptimizeB(Grad grad)
        {
            throw new NotImplementedException();
        }
        //To be done.

        public double Predict(Grad grad, double[] inputs)
        {
            int size = inputs.Length;
            double[] feature_calcs = new double[size];
            for (int i = 0; i < size; i++)
            {
                Wopt wop = grad.ws[i];
                double feature_calc = wop.w * inputs[i];
                feature_calcs[i] = feature_calc;
            }
            return feature_calcs.Sum() + grad.b.b;
        }

        public double[] InputDerivatives(Grad grad, double[] inputs)
        {
            int size = inputs.Length;
            double[] input_derivs = new double[size];

            for (int i = 0; i < size; i++)
            {
                double input_deriv = grad.derivs[i] * inputs[i];
                input_derivs[i] = input_deriv;
            }

            return input_derivs;
        }

        public double[] ErrorDerivatives(Grad grad, double[] targets)
        {
            int size = targets.Length;
            double[] error_derivs = new double[size];

            for (int i = 0; i < size; i++)
            {
                double error_deriv = 2 * (grad.preds[i] - targets[i]);
                error_derivs[i] = error_deriv;
            }

            return error_derivs;
        }

        public double[] Errors(Grad grad, double[] targets)
        {
            int size = targets.Length;
            double[] errors = new double[size];

            for(int i = 0; i < size; i++)
            {
                double error = Math.Pow(grad.preds[i] - targets[i], 2);
                errors[i] = error;
            }

            grad.error = errors.Sum() / (2 * targets.Length);
            return errors;
        }

        public double[] Predictions(Grad grad, double[][] inputs)
        {
            int o_size = inputs.Length;
            int i_size = inputs[0].Length;
            double[] predictions = new double[o_size];
            for (int i = 0; i < o_size; i++)
            {
                double[] features = inputs[i];
                double[] feature_calcs = new double[i_size];
                for (int j = 0; j < i_size; j++)
                {
                    Wopt wopt = grad.ws[j];
                    double feature_calc = wopt.w * features[j];
                    feature_calcs[j] = feature_calc;
                }
                double prediction = feature_calcs.Sum() + grad.b.b;
                predictions[i] = prediction;
            }
            return predictions;
        }


        public void Train(Grad grad, MMODEL[][] batches, bool optim_activated)
        {
            Adam opt = new Adam();
            grad.UpdateW(batches[0][0].input);
            Transposer tr = new Transposer();

            double w_var;
            double b_var;

            while (grad.error >= 0)
            {
                grad.epoch++;
                for (grad.bid = 0; grad.bid < batches.Length; grad.bid++)
                {
                    MMODEL[] batch = batches[grad.bid];
                    double[][] inputs = batch.Select(x => x.input).ToArray();
                    double[] targets = batch.Select(x => x.target).ToArray();
                    grad.d = batch.Length;

                    grad.preds = Predictions(grad, inputs);
                    grad.errors = Errors(grad, targets);
                    grad.derivs = ErrorDerivatives(grad, targets);

                    double[][] inputsT = tr.TransposeList(inputs);
                    for (grad.fid = 0; grad.fid < inputsT.Length; grad.fid++)
                    {
                        Wopt wop = grad.ws[grad.fid];
                        grad.input_derivs = InputDerivatives(grad, inputsT[grad.fid]);

                        if (!optim_activated)
                            w_var = wop.w - grad.a * grad.GetJW();
                        else
                            w_var = opt.Optimize(grad, false);
                        wop.w = w_var;
                    }

                    Bopt bop = grad.b;

                    if (!optim_activated)
                        b_var = bop.b - grad.a * grad.GetJB();
                    else
                        b_var = opt.Optimize(grad, true);
                    bop.b = b_var;

                    if (grad.error <= Math.Pow(10, -2))
                        break;
                }
                if (grad.error <= Math.Pow(10, -2))
                    break;
            }
        }
    }
}

