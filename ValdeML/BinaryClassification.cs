using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ValdeML
{
    internal class BCS : iBCSF
    {
        public double Prediction(Grad grad, double input)
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
            for(int i= 0; i< inputs.Length; i++)
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
            for(int i = 0; i< targets.Length; i++)
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
            for(int i = 0; i< inputs.Length; i++)
            {
                double prediction = (grad.w * inputs[i]) + grad.b;
                double activation = SigmoidActivation(prediction);
                predictions[i] = activation;
            }
            return predictions;
        }
        public void Train(Grad grad, SMODEL[][] batches)
        {
            while(grad.error >= 0)
            {
                grad.epoch++;
                if (grad.keep_training)
                {
                    for(grad.bid= 0; grad.bid< batches.Length; grad.bid++)
                    {
                        SMODEL[] batch = batches[grad.bid];
                        grad.d = batch.Length;

                        double[] inputs = batch.Select(x => x.input).ToArray();
                        double[] targets = batch.Select(x => x.target).ToArray();

                        grad.preds = Predictions(grad, inputs);
                        grad.errors = Errors(grad, targets);
                        grad.derivs = ErrorDerivatives(grad, targets);
                        grad.input_derivs = InputDerivatives(grad, inputs);

                        //double tmp_w = grad.w - (grad.a * grad.GetJW());
                        double tmp_w = grad.w - (grad.a * OptimizeW(grad));
                        grad.w = tmp_w;

                        //double tmp_b = grad.b - (grad.a * grad.GetJB());
                        double tmp_b = grad.b - (grad.a * OptimizeB(grad));
                        grad.b = tmp_b;

                        grad.GetError();

                        //if (grad.error <= Math.Pow(10, -3))
                        //    break;
                    }
                    if (grad.error <= Math.Pow(10, -3))
                        break;
                }
                else
                    break;
            }
        }
    }
}
