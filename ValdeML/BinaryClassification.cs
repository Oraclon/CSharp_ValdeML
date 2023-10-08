using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ValdeML
{
    internal class BCS : iBCSF
    {
        public double Predict(Grad grad, double input)
        {
            double prediction = grad.w * input + grad.b;
            double activation = SigmoidActivation(prediction);
            return activation;
        }
        //Experimental
        public double OptimizeW(Grad grad)
        {
            Wopt wop = grad.wop;
            double old_vdw = grad.b1 * wop.vdw + (1 - grad.b1) * grad.GetJW();
            double vdw_c = old_vdw / (1 - Math.Pow(grad.b1, grad.d));
            wop.vdw = vdw_c;

            double old_sdw = grad.b2 * wop.sdw + (1 - grad.b2) * Math.Pow(grad.GetJW(), 2);
            double sdw_c = old_sdw / (1 - Math.Pow(grad.b2, grad.d));
            wop.sdw = sdw_c;

            //return wop.vdw/Math.Sqrt(wop.sdw)+ grad.e;
            return wop.vdw;
        }
        public double OptimizeB(Grad grad)
        {
            Bopt bop = grad.bop;
            double old_vdb = grad.b1 * bop.vdb + (1 - grad.b1) * grad.GetJB();
            double vdb_c = old_vdb / (1 - Math.Pow(grad.b1, grad.d));
            bop.vdb = vdb_c;

            double old_sdb = grad.b2 * bop.sdb + (1 - grad.b2) * Math.Pow(grad.GetJB(), 2);
            double sdb_c = old_sdb / (1 - Math.Pow(grad.b2, grad.d));
            bop.sdb = sdb_c;

            //return bop.vdb/ Math.Sqrt(bop.sdb)+ grad.e;
            return bop.vdb;
        }
        //Experimental
        public double[] InputDerivatives(Grad grad, double[] inputs)
        {
            double[] input_derivatives = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                double input_derivative = grad.derivs[i] * inputs[i];
                input_derivatives[i] = input_derivative;
            }
            return input_derivatives;
        }
        public double[] ErrorDerivatives(Grad grad, double[] targets)
        {
            double[] error_derivatives = new double[targets.Length];
            for (int i = 0; i < targets.Length; i++)
            {
                double error_derivative = 2 * (grad.preds[i] - targets[i]);
                double pred_derivative = grad.preds[i] * (1 - grad.preds[i]);
                //double derivative = (pred_derivative * error_derivative) / targets.Length;
                double derivative = (error_derivative * pred_derivative) / targets.Length;
                error_derivatives[i] = derivative;
            }
            return error_derivatives;
        }
        public double[] Errors(Grad grad, double[] targets)
        {
            double[] errors = new double[targets.Length];
            for (int i = 0; i < targets.Length; i++)
            {
                double error = Math.Pow(grad.preds[i] - targets[i], 2) / (2 * targets.Length);
                errors[i] = error;
            }
            return errors;
        }
        public double SigmoidActivation(double prediction)
        {
            return 1.0 / (1 + Math.Exp(-1.0 * prediction));
        }
        public double[] Predictions(Grad grad, double[] inputs)
        {
            double[] predictions = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                double prediction = (grad.w * inputs[i]) + grad.b;
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
                if (grad.keep_training)
                {
                    for (grad.bid = 0; grad.bid < batches.Length; grad.bid++)
                    {
                        SMODEL[] batch = batches[grad.bid];
                        grad.d = batch.Length;

                        double[] inputs = batch.Select(x => x.input).ToArray();
                        double[] targets = batch.Select(x => x.target).ToArray();

                        grad.preds = Predictions(grad, inputs);
                        grad.errors = Errors(grad, targets);
                        grad.derivs = ErrorDerivatives(grad, targets);
                        grad.input_derivs = InputDerivatives(grad, inputs);

                        double tmp_w = grad.w - (grad.a * grad.GetJW());
                        //double tmp_w = grad.w - (grad.a * OptimizeW(grad));
                        grad.w = tmp_w;

                        double tmp_b = grad.b - (grad.a * grad.GetJB());
                        //double tmp_b = grad.b - (grad.a * OptimizeB(grad));
                        grad.b = tmp_b;

                        grad.GetError();

                        if (grad.error <= Math.Pow(10, -3))
                            grad.keep_training = !grad.keep_training;
                    }
                }
                else
                    break;
            }
        }
    }
    internal class BCM : iBCMF
    {
        public double Predict(Grad grad, double[] inputs)
        {
            int size = inputs.Length;
            double[] feat_preds = new double[size];
            for (int i = 0; i < size; i++)
            {
                feat_preds[i] = grad.ws[i] * inputs[i];
            }
            double pred = feat_preds.Sum() + grad.b;
            return SigmoidActivation(pred);
        }

        public double OptimizeB(Grad grad)
        {
            Bopt bop = grad.bop;
            double old_vdb = grad.b1 * bop.vdb + (1 - grad.b1) * CalcJB(grad);
            bop.vdb = old_vdb;
            return bop.vdb;
        }
        public double OptimizeW(Grad grad)
        {
            Wopt wop = grad.wops[grad.fid];
            double old_vdw = grad.b1 * wop.vdw + (1 - grad.b1) * CalcJW(grad);
            wop.vdw = old_vdw;
            return wop.vdw;
        }

        double CalcJW(Grad grad)
        {
            double[] input_derivs = grad.input_derivs;
            int size = input_derivs.Length;
            return input_derivs.Sum() / size;
            //return input_derivs.Sum() ;
        }
        double CalcJB(Grad grad)
        {
            double[] derivs = grad.derivs;
            int size = derivs.Length;
            return derivs.Sum() / size;
            //return derivs.Sum() ;
        }
        void CalcError(Grad grad)
        {
            double[] errors = grad.errors;
            int size = errors.Length;
            grad.error = errors.Sum() / size;
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
            double[] error_derivatives = new double[size];
            for (int i = 0; i < size; i++)
            {
                double error = grad.errors[i];
                double derivative = grad.preds[i] - targets[i];
                //double derivative = targets[i] == 1 ? -1 / grad.preds[i] : 1 / (1 - grad.preds[i]);
                double activat_der = grad.preds[i] * (1 - grad.preds[i]);
                double error_derivative = derivative * activat_der;
                error_derivatives[i] = error_derivative;
            }
            return error_derivatives;
        }
        public double[] Errors(Grad grad, double[] targets)
        {
            int size = targets.Length;
            double[] errors = new double[size];
            for (int i = 0; i < size; i++)
            {
                double y = grad.preds[i];
                double t = targets[i];
                double error = -t * Math.Log(y) - (1 - t) * Math.Log(1 - y);
                errors[i] = error;
            }
            return errors;
        }

        public double SigmoidActivation(double prediction)
        {
            return 1.0 / (1 + Math.Exp(-prediction));
        }
        public double[] Predictions(Grad grad, double[][] inputs)
        {
            int outs = inputs.Length;
            int inps = inputs[0].Length;

            double[] predictions = new double[outs];

            for (int i = 0; i < outs; i++)
            {
                double[] feat_calcs = new double[inps];
                double[] input = inputs[i];
                for (int j = 0; j < inps; j++)
                {
                    double feat_calc = grad.ws[j] * input[j];
                    feat_calcs[j] = feat_calc;
                }
                double prediction = feat_calcs.Sum() + grad.b;
                double activation = SigmoidActivation(prediction);
                predictions[i] = activation;
            }
            return predictions;
        }

        public void Train(Grad grad, MMODEL[][] batches)
        {
            grad.UpdateW(batches[0][0].input);
            Transposer tr = new Transposer();
            while (grad.error >= 0)
            {
                grad.epoch++;
                if (grad.keep_training)
                {
                    for (grad.bid = 0; grad.bid < batches.Length; grad.bid++)
                    {
                        MMODEL[] batch = batches[grad.bid];
                        grad.d = batch.Length;
                        double[][] inputs = batch.Select(x => x.input).ToArray();
                        double[] targets = batch.Select(x => x.target).ToArray();

                        grad.preds = Predictions(grad, inputs);
                        grad.errors = Errors(grad, targets);
                        grad.derivs = ErrorDerivatives(grad, targets);

                        double[][] inputsT = tr.TransposeList(inputs);
                        for (grad.fid = 0; grad.fid < inputsT.Length; grad.fid++)
                        {
                            double[] inputT = inputsT[grad.fid];
                            grad.input_derivs = InputDerivatives(grad, inputT);

                            double tmp_w = grad.ws[grad.fid] - grad.a * grad.GetJW();
                            //double tmp_w = grad.ws[grad.fid] - grad.a * CalcJW(grad);
                            //double tmp_w = grad.ws[grad.fid] - grad.a * OptimizeW(grad);
                            grad.ws[grad.fid] = tmp_w;
                        }

                        double tmp_b = grad.b - grad.a * grad.GetJB();
                        //double tmp_b = grad.b - grad.a * CalcJB(grad);
                        //double tmp_b = grad.b - grad.a * OptimizeB(grad);
                        grad.b = tmp_b;

                        CalcError(grad);
                    }
                    if (grad.error <= Math.Pow(10, -2))
                        break;
                }
                else
                {
                    break;
                }
                Console.WriteLine(grad.error);
            }
        }
    }
}
