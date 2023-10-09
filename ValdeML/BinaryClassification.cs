using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ValdeML
{
    public class BCS : iBCSF
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
            double prediction = grad.w.w * input + grad.b.b;
            return (int)Math.Round(SigmoidActivation(prediction));
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
                double e_deriv = targets[i] == 1 ? -1 / grad.preds[i] : 1 / (1 - grad.preds[i]);
                double p_deriv = grad.preds[i] * (1 - grad.preds[i]);
                double deriv = e_deriv * p_deriv;
                error_derivs[i] = deriv;
            }

            return error_derivs;
        }

        public double[] Errors(Grad grad, double[] targets)
        {
            int size = targets.Length;
            double[] errors = new double[size];

            for (int i = 0; i < size; i++)
            {
                double error = targets[i] == 1 ? -Math.Log(grad.preds[i]) : -Math.Log(1 - grad.preds[i]);
                errors[i] = error;
            }

            grad.error = errors.Sum() / size;
            return errors;
        }

        public double SigmoidActivation(double prediction)
        {
            return 1 / (1 + Math.Exp(-prediction));
        }

        public double[] Predictions(Grad grad, double[] inputs)
        {
            int size = inputs.Length;
            double[] predictions = new double[size];

            for (int i = 0; i < size; i++)
            {
                Wopt wop = grad.w;
                Bopt bop = grad.b;

                double prediction = wop.w * inputs[i] + bop.b;
                double activation = SigmoidActivation(prediction);
                predictions[i] = activation;
            }

            return predictions;
        }


        public void Train(Grad grad, SMODEL[][] batches)
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
                    double tmp_w = wop.w - grad.a * grad.GetJW();
                    wop.w = tmp_w;

                    Bopt bop = grad.b;
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

    public class BCM : iBCMF
    {
        //To be done.
        public double OptimizeB(Grad grad)
        {
            throw new NotImplementedException();
        }

        public double OptimizeW(Grad grad)
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

            double prediction = feature_calcs.Sum() + grad.b.b;
            return (int)Math.Round(SigmoidActivation(prediction));
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
                double e_deriv = targets[i] == 1 ? -1 / grad.preds[i] : 1 / (1 - grad.preds[i]);
                double p_deriv = grad.preds[i] * (1 - grad.preds[i]);
                double error = e_deriv * p_deriv;
                error_derivs[i] = error;
            }

            return error_derivs;
        }

        public double[] Errors(Grad grad, double[] targets)
        {
            int size = targets.Length;
            double[] errors = new double[size];

            for(int i = 0; i< size; i++)
            {
                double error = targets[i] == 1 ? -Math.Log(grad.preds[i]) : -Math.Log(1 - grad.preds[i]);
                errors[i] = error;
            }

            grad.error = errors.Sum() / size;
            return errors;
        }

        public double SigmoidActivation(double prediction)
        {
            return 1 / (1 + Math.Exp(-prediction));
        }

        public double[] Predictions(Grad grad, double[][] inputs)
        {
            int o_size = inputs.Length;
            int i_size = inputs[0].Length;

            double[] predictions = new double[o_size];

            for (int i = 0; i < o_size; i++)
            {
                double[] feature = inputs[i];
                double[] feature_calcs = new double[i_size];
                for (int j = 0; j < i_size; j++)
                {
                    Wopt wop = grad.ws[j];
                    double feature_calc = wop.w * feature[j];
                    feature_calcs[j] = feature_calc;
                }
                double prediction = feature_calcs.Sum() + grad.b.b;
                double activation = SigmoidActivation(prediction);
                predictions[i] = activation;
            }

            return predictions;
        }


        public void Train(Grad grad, MMODEL[][] batches)
        {
            Adam opt = new Adam();
            grad.UpdateW(batches[0][0].input);
            Transposer tr = new Transposer();
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

                        double tmp_w = wop.w - grad.a * grad.GetJW();
                        wop.w = tmp_w;
                        //wop.w = opt.Optimize(grad, false);
                    }

                    Bopt bop = grad.b;
                    double tmp_b = bop.b - grad.a * grad.GetJB();
                    bop.b = tmp_b;
                    //bop.b = opt.Optimize(grad, true);

                    if (grad.error <= Math.Pow(10, -2))
                        break;
                }
                if (grad.error <= Math.Pow(10, -2))
                    break;
            }
        }
    }
}
