using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ValdeML
{
    internal class BCS : iBCSF
    {
        public double[] Prediction(Grad grad, double input)
        {
            throw new NotImplementedException();
        }
        //Experimental
        public double OptimizeW(Grad grad)
        {
            throw new NotImplementedException();
        }
        public double OptimizeB(Grad grad)
        {
            throw new NotImplementedException();
        }
        //Experimental
        public double[] InputDerivatives(Grad grad, double[] inputs)
        {
            double[] input_derivatives = new double[inputs.Length];

            return input_derivatives;
        }
        public double[] ErrorDerivatives(Grad grad, double[] targets)
        {
            double[] error_derivatives = new double[targets.Length];

            return error_derivatives;
        }
        public double[] Errors(Grad grad, double[] targets)
        {
            double[] errors = new double[targets.Length];
            
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
                double prediction = grad.w * inputs[i] + grad.b;
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

                        grad.GetError();
                    }
                }
                else
                    break;
            }
        }
    }
}
